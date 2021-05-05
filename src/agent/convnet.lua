--[[
Copyright (c) 2014 Google Inc.

See LICENSE file for full terms of limited license.
]]

require "initenv"

function create_network(args)


	local net = nn.Sequential()
	
	
    local conv_layers = nn.Sequential()
    conv_layers:add(nn.Reshape(unpack(args.input_dims)))

    print(args.input_dims)
    print(unpack(args.input_dims))

    for k, v in pairs(args.input_dims) do
        print(v)
    end


    --- first convolutional layer
    local convLayer = nn.SpatialConvolution

    conv_layers:add(convLayer(args.hist_len*args.ncols, args.n_units[1],
                        args.filter_size[1], args.filter_size[1],
                        args.filter_stride[1], args.filter_stride[1],1))
    conv_layers:add(args.nl())

    -- Add convolutional layers
    for i=1,(#args.n_units-1) do
        -- second convolutional layer
        conv_layers:add(convLayer(args.n_units[i], args.n_units[i+1],
                            args.filter_size[i+1], args.filter_size[i+1],
                            args.filter_stride[i+1], args.filter_stride[i+1]))
        conv_layers:add(args.nl())
    end

    local nel
    if args.gpu >= 0 then
        nel = conv_layers:cuda():forward(torch.zeros(1,unpack(args.input_dims))
                :cuda()):nElement()
    else
        nel = conv_layers:forward(torch.zeros(1,unpack(args.input_dims))):nElement()
    end

    -- reshape all feature planes into a vector per example
    conv_layers:add(nn.Reshape(nel))
	

	local image_size = 4 * 84 * 84
	
	local concat = nn.ConcatTable()
    concat:add(nn.Sequential():add(nn.Narrow(2, 1, image_size)):add(conv_layers))
    concat:add(nn.Sequential():add(nn.Narrow(2, image_size + 1, args.max_nodes)):add(nn.Identity()))
	
	
    net:add(nn.Reshape(4 * 84 * 84 + args.max_nodes))
	net:add(concat)
    net:add(nn.JoinTable(2))
	
	
    -- fully connected layer
    net:add(nn.Linear(nel + args.max_nodes, args.n_hid[1]))
    net:add(args.nl())
    local last_layer_size = args.n_hid[1]

    for i=1,(#args.n_hid-1) do
        -- add Linear layer
        last_layer_size = args.n_hid[i+1]
        net:add(nn.Linear(args.n_hid[i], last_layer_size))
        net:add(args.nl())
    end

    -- add the last fully connected layer (to actions)
    net:add(nn.Linear(last_layer_size, args.n_actions))

    if args.gpu >=0 then
        net:cuda()
    end
    if args.verbose >= 2 then
        print(net)
        print('Convolutional layers flattened output size:', nel)
    end
    return net
end
