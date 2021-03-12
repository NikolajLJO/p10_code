--[[
Copyright (c) 2014 Google Inc.

See LICENSE file for full terms of limited license.
]]

require "initenv"

function create_network(args)

    local encoder = nn.Sequential()

    --encoder:add(nn.Reshape(unpack(args.input_dims)))
    encoder:add(nn.Reshape(1, 84, 84))

    --- first convolutional layer
    local convLayer = nn.SpatialConvolution

    encoder:add(convLayer(1, args.n_units[1],
                        args.filter_size[1], args.filter_size[1],
                        args.filter_stride[1], args.filter_stride[1],1))

    encoder:add(args.nl())

    -- Add convolutional layers
    for i = 1, (#args.n_units - 1) do
        encoder:add(convLayer(args.n_units[i], args.n_units[i + 1],
                            args.filter_size[i + 1], args.filter_size[i + 1],
                            args.filter_stride[i + 1], args.filter_stride[i + 1]))

        encoder:add(args.nl())
    end

    local nel
    if args.gpu >= 0 then
        --nel = encoder:cuda():forward(torch.zeros(1, unpack(args.input_dims)):cuda()):nElement()
        nel = encoder:cuda():forward(torch.zeros(1, 1, 84, 84):cuda()):nElement()
    else
        --nel = encoder:forward(torch.zeros(1, unpack(args.input_dims))):nElement()
        nel = encoder:forward(torch.zeros(1, 1, 84, 84)):nElement()
    end

    -- Reshape all feature planes into a vector per example
    encoder:add(nn.Reshape(nel))


    -- The siamese model
    local siamese_encoder = nn.ParallelTable()
    siamese_encoder:add(encoder)
    siamese_encoder:add(encoder:clone('weight', 'bias', 'gradWeight', 'gradBias', 'running_mean', 'running_std', 'running_var'))


    local net = nn.Sequential()
    net:add(nn.SplitTable(2)) -- split input tensor along the rows (1st dimension) to table for input to ParallelTable
    net:add(siamese_encoder)


    local mean = nn.Sequential()
    mean:add(nn.CAddTable())
    mean:add(nn.MulConstant(0.5))
    
    local subAndMean = nn.ConcatTable()
    subAndMean:add(nn.CSubTable())
    subAndMean:add(mean)
    net:add(subAndMean)

    net:add(nn.JoinTable(2))

    local fac = 1
    net:add(nn.CustomLinear(2 * nel, args.n_hid[1], 0, math.sqrt(fac * 2 / (2 * nel))))

    net:add(args.nl())

    local final_hidden_units = args.n_hid[1]

    for i = 2, #(args.n_hid) do

        final_hidden_units = args.n_hid[i]

        net:add(nn.CustomLinear(args.n_hid[i - 1], args.n_hid[i], 0, math.sqrt(2 / (args.n_hid[i - 1]))))

        net:add(args.nl())
    end


    local scalingFactor = 1
    net:add(nn.CustomLinear(final_hidden_units, #args.actions, 0, scalingFactor * math.sqrt(2 / (final_hidden_units))))

    -- Ensure outputs sum to zero
    --[[
    local copyMean = nn.Sequential()
    copyMean:add(nn.Mean(2))
    copyMean:add(nn.Replicate(#args.actions))
    copyMean:add(nn.Transpose({1, 2}))

    local con = nn.ConcatTable()
    con:add(nn.Identity())
    con:add(copyMean)

    net:add(con)
    net:add(nn.CSubTable())
    --]]

    if args.gpu >=0 then
        net:cuda()
    end
    if args.verbose >= 2 then
        print(net)
        print('Convolutional layers flattened output size:', nel)
    end
    return net
end