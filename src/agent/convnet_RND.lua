require "initenv"

function create_network(args)
    
    local prednet = nn.Sequential()

    -- network input
    conv_layers = nn.Sequential()
    conv_layers:add(nn.reshape(1,84,84))
    
    -- first convlayer
    conv_layers:add(nn.SpatialConvolution(1, args.n_units[1],
                                          args.filter_size[1], 
                                          args.filter_size[1],
                                          args.filter_stride[1],
                                          args.filter_stride[1],1))
    conv_Layers:add(nn.Rectifier())

    -- additional convlayers
    for i=1,(#args.n_units-1) do
        conv_layers:add(nn.SpatialConvolution(args.n_units[i],
                                              args.n_units[i+1],
                                              args.filter_size[i+1], 
                                              args.filter_size[i+1],
                                              args.filter_stride[i+1],
                                              args.filter_stride[i+1]))
        conv_Layers:add(nn.Rectifier())
    end

    local nel
    if args.gpu >= 0 then
        --nel = encoder:cuda():forward(torch.zeros(1, unpack(args.input_dims)):cuda()):nElement()
        nel = encoder:cuda():forward(torch.zeros(1, 1, 84, 84):cuda()):nElement()
    else
        --nel = encoder:forward(torch.zeros(1, unpack(args.input_dims))):nElement()
        nel = encoder:forward(torch.zeros(1, 1, 84, 84)):nElement()
    end

    conv_Layers:add(nn.Reshape(nel))
    
    -- MLP
    net:add(nn.Linear(nel, args.n_hid[1]))
    net:add(nn.Rectifier())
    local last_layer_size = args.n_hid[1]

    -- additional linearlayers
    for i=1,(#args.n_hid-1) do
        last_layer_size = args.n_hid[i+1]
        net:add(nn.Linear(args.n_hid[i], last_layer_size))
        net:add(nn.Rectifier())
    end

    prednet:add(nn.Linear(last_layer_size, args.rnd_out) 

    print(prednet)

    return 1

end
