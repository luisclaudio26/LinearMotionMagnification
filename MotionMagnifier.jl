module MotionMagnifier
using Images

tmp_dir = "./read_tmp"
Frame64 = Matrix{Float64}
FrameBlock = Array{Frame64,1}

#################################
########### VIDEO IO ############
#################################
# Is it possible to compress this into a single tensor operation?
function YIQtoRGB(Y::Frame64, I::Frame64, Q::Frame64)
    R = Y + 0.956.*I + 0.621.*Q
    G = Y - 0.272.*I - 0.647.*Q
    B = Y - 1.106.*I + 1.703.*Q
    return R, G, B
end

function RGBtoYIQ(R::Frame64, G::Frame64, B::Frame64)
    Y = 0.299.*R + 0.587.*G + 0.114.*B
    I = 0.596.*R - 0.274.*G - 0.322.*B
    Q = 0.211.*R - 0.523.*G + 0.312.*B
    return Y, I, Q
end

function read_video(path::String)::Tuple{Int64, Tuple{Int64,Int64}}
    if !isdir(tmp_dir)
        mkdir(tmp_dir)
    end
    
    # start by opening the video and saving
    # individual frames to .png files. Fortunately
    # ffmpeg does this for us
    run(`ffmpeg -i $path $tmp_dir/tmp_%05d.png`)
    
    # count the number of frames read by counting the number of lines
    # output by "ls -l" inside read_tmp: should be one for each frame
    # and one more for the "header"
    n_frames = parse(Int64, readstring( pipeline(`ls -l $tmp_dir/`, `wc -l`) ))

    # load one frame so to compute video dimensions
    y, i, q = grab_frame_from_video(1, n_frames-1)
    
    return n_frames-1, size(y)
end

function grab_frame_from_video(s::Int64, n_frames::Int64)
    @assert s <= n_frames
    
    # the image files are named tmp_0000.png,
    # tmp_0001.png, etc.
    # TODO: this will break if we change tmp_dir!!!
    fname = @sprintf "./read_tmp/tmp_%05d.png" s;
            
    # Convert to YIQ space. Only the Y matter for the
    # processing, but in the end we need to compose it back
    # with the I and Q channels before converting back to RGB
    rgb = channelview(load(fname))
    Y, I, Q = RGBtoYIQ(Float64.(rgb[1,:,:]), Float64.(rgb[2,:,:]), Float64.(rgb[3,:,:]))
    return Y, I, Q
end

function output_single_frame(f::Frame64, index::Int64, path::String, clamp_vals = true)
    frameID = string(path, "/frame", index, ".png")  
    frame = clamp_vals ? (f.+1.0)*0.5 : f
    frame = clamp01nan.(frame)
    save(frameID, frame)
end

function output_single_frame(f::Tuple{Frame64,Frame64,Frame64}, index::Int64, 
                                path::String, clamp_vals = true)
    frameID = string(path, "/frame", index, ".png")  
    
    r_ = clamp_vals ? (f[1].+1.0)*0.5 : f[1]
    g_ = clamp_vals ? (f[2].+1.0)*0.5 : f[2]
    b_ = clamp_vals ? (f[3].+1.0)*0.5 : f[3]
    frame = colorview(RGB, clamp01nan.(r_), 
                           clamp01nan.(g_), 
                           clamp01nan.(b_))
    save(frameID, frame)
end

function build_video(name::String, path::String, fps::Float64)
    run(`ffmpeg -loglevel -8 -y -framerate $fps -i $path/frame%d.png $path/$name.mp4`)
end

###########################
######### UTILITY #########
###########################
mutable struct RingBuffer
    frame_block::FrameBlock
    current::Int64
    discard::Int64
    blocksz::Int64    
    dim::Tuple{Int64,Int64}

    function RingBuffer(support::Int64, initial_data::FrameBlock)
        # we work with filter of odd-length support only! 
        @assert length(support) % 2 == 1
        @assert length(initial_data) == support
    
        # initialize ring buffer settings
        # TODO: if frames do not have all the same size,
        # everything will crash! We should do some
        # checking
        dim = size(initial_data[1])
        blocksz = support
        frame_block = FrameBlock(blocksz)
    
        # copy initialization data
        for i in 1:blocksz
            frame_block[i] = copy( initial_data[i] )
        end

        # remember that discard and block_current
        # must be zero-index based so we can use
        # modular arithmetic
        discard = 0
        current = blocksz ÷ 2
    
        return new(frame_block, current, discard, blocksz, dim)
    end
end

function advance_block(rb::RingBuffer, new_frame::Frame64)
    # grab new frame and place it on the next
    # frame to be discarded
    rb.frame_block[rb.discard+1] = copy(new_frame)

    # (circularly) advance pointers
    rb.current = (rb.current + 1) % rb.blocksz
    rb.discard = (rb.discard + 1) % rb.blocksz
end

#######################
####### FILTERS #######
#######################
function identity_1(f::Vector{Float64})
    @assert length(f) == 1
    return f[1]
end

function box_5(f::Vector{Float64})
    @assert length(f) == 5
    return sum(f)/5
end

#############################################
####### EULERIAN MOTION MAGNIFICATION #######
#############################################
function temporal_filter(rb::RingBuffer, 
                            block::FrameBlock, 
                            filter::Function, 
                            frame0::Frame64)::Frame64
    out = Frame64(rb.dim...)
    
    # for each pixel position, collect time-series
    # (i.e. the pixel values in the same positions
    # across the block) and filter it
    for i in 1:rb.dim[1], j in 1:rb.dim[2]
        
        time_series = Vector{Float64}(rb.blocksz)
        
        # remember that block is a circular buffer and
        # we wanna take n÷2 elements to the left and to the
        # right, thus we need to get the index of the central
        # element (vp.block_current), take the offset to the
        # element we want (..., -2, -1, 0, 1, 2, ...) and wrap
        # ir around the size of the block. For example:
        #
        # block = [9 10 6 7 8]
        # block_current = 4
        #
        # so we want to take 2 elements to the left of the
        # (4+1)th element (frame 8) and 2 to the right.
        #
        # 4-2 mod 5 = 2 -> frame 6
        # 4-1 mod 5 = 3 -> frame 7
        # 4+0 mod 5 = 4 -> frame 8
        # 4+1 mod 5 = 0 -> frame 9
        # 4+2 mod 5 = 1 -> frame 10
        #
        # we store the frames in this order inside time_series
        # so we can safely apply a filter to it now
        sup_radius = rb.blocksz ÷ 2 ; idx = 1
        for k in -sup_radius : sup_radius
            # modular arithmetic in Julia defines the modulus
            # of a negative number as -(|x mod N|), i.e., modulus
            # operator won't wrap around to positive numbers when
            # argument is negative. If this is the case, we fix by
            # computing N - |(x mod N)| = N + (x mod N).
            p = (rb.current+k) % rb.blocksz
            if( p < 0 ) p = rb.blocksz + p end
            
            time_series[idx] = block[p+1][i,j]
            idx = idx + 1
        end
        
        # apply Wadhwah's non-linear filtering
        out[i,j] = filter(time_series) - frame0[i,j]
    end
    
    return out
end

function process_frame(rb::RingBuffer, 
                        α::Float64, n_laplacian::Int64, 
                        filter::Function, filter_support::Int64, 
                        frame0::Frame64)
    out = zeros(rb.dim...)
    last_level = copy(rb.frame_block)
    f0 = copy(frame0)
    blurred = FrameBlock(rb.blocksz)    
    
    for i in 1:n_laplacian
        
        band = last_level
        f0_band = f0
        
        # if we haven't reached the last level of the pyramid,
        # we need to compute a blurred version of last_level
        # and subtract it to obtain the bandpassed version.
        # if i == n_laplacian, this last_level contains the
        # lowpassed residual and we do nothing
        if i < n_laplacian
            # filter each frame inside block individually AND frame0
            # TODO: experiment with other filters and supports!!!
            f0_g = imfilter(f0, Kernel.gaussian(2))
            
            # TODO: why can't I just use imfilter.()?
            for f in 1:rb.blocksz
                blurred[f] = imfilter(last_level[f], Kernel.gaussian(2))
            end
            
            # band stores the last level, so doing
            # band - blurred computes the current Laplacian
            # level we want
            band = band .- blurred
            f0_band = f0 .- f0_g
            
            # for the next iteration, our last_level is the
            # blured version of the current level
            last_level .= blurred
            f0 .= f0_g
        end
        
        # band now has the bandpassed frame block and we
        # can temporally filter it
        filtered_band = temporal_filter(rb, band, filter, f0_band)
        
        # accumulate to the final frame we're composing
        out += band[rb.current+1] + α .* filtered_band
    end
    
    return out
end

function process_video(frame_grabber::Function, n_frames::Int64,
                        α::Float64, n_laplacian::Int64, 
                        filter::Function, filter_support::Int64,
                        path::String)
    # target path
    if(!isdir(path)) mkdir(path) end
    
    # we won't deal with border cases
    n_frames_out = n_frames - (filter_support - 1)
    
    # preload as many frames as needed 
    # for the ring buffers initialization
    initY = FrameBlock(filter_support)
    initI = FrameBlock(filter_support)
    initQ = FrameBlock(filter_support)
    for i in 1:filter_support
        initY[i], initI[i], initQ[i] = frame_grabber(i)
    end
    
    # create ring buffers
    Y = RingBuffer(filter_support, initY)
    I = RingBuffer(filter_support, initI)
    Q = RingBuffer(filter_support, initQ)
    
    # force cleanup to save memory
    initY, initI, initQ = nothing, nothing, nothing
    gc()
    
    # in particular, we need frame ZERO for motion magnification
    frame0 = Y.frame_block[1]
    
    # main loop: process -> output -> load frame -> advance.
    # we'll need an extra step after this loop for the last frame
    for i in 1:(n_frames_out-1)
        y_ = process_frame(Y, α, n_laplacian, filter, filter_support, frame0)
        
        # TODO: rebuild three-channel frame before outputting it
        r, g, b = YIQtoRGB(y_, I.frame_block[I.current+1], Q.frame_block[Q.current+1])
        output_single_frame((r,g,b), i, path, false)
        
        # load next and advance buffers
        next_y, next_i, next_q = frame_grabber(filter_support+i)
        advance_block(Y, next_y)
        advance_block(I, next_i)
        advance_block(Q, next_q)
    end
    
    #TODO: process last frame
end

end