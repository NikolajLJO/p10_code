require 'torch'
require 'nn'
require 'nngraph'
require 'nnutils'
--require 'more_utils'
--require 'image'
--require 'optim'
--require 'Scale'
--require 'NeuralQLearner'
--require 'TransitionTable'
--require 'Node'
--require 'Rectifier'
--require 'ALEWrapper'
--require 'CustomLinear'
--require 'DebugWindow'

function create_network(args)
    
    

    -- network input
    local conv_layers = nn.Sequential()
    conv_layers:add(nn.Reshape(2,84,84))
    
    -- first convlayer
    conv_layers:add(nn.SpatialConvolution(2, args.n_units[1],
                                          args.filter_size[1], 
                                          args.filter_size[1],
                                          args.filter_stride[1],
                                          args.filter_stride[1],1))
    conv_layers:add(nn.Rectifier())

    -- additional convlayers
    for i=1,(#args.n_units-1) do
        conv_layers:add(nn.SpatialConvolution(args.n_units[i],
                                              args.n_units[i+1],
                                              args.filter_size[i+1], 
                                              args.filter_size[i+1],
                                              args.filter_stride[i+1],
                                              args.filter_stride[i+1]))
        conv_layers:add(nn.Rectifier())
    end

    local nel
    if args.gpu >= 0 then
        --nel = encoder:cuda():forward(torch.zeros(1, unpack(args.input_dims)):cuda()):nElement()
        nel = conv_layers:cuda():forward(torch.zeros(1, 2, 84, 84):cuda()):nElement()
    else
        --nel = encoder:forward(torch.zeros(1, unpack(args.input_dims))):nElement()
        nel = conv_layers:forward(torch.zeros(1, 2, 84, 84)):nElement()
    end
    
    
    conv_layers:add(nn.Reshape(nel))
    
    local parallelle_conv_layers = nn.ParallelTable()
    parallelle_conv_layers:add(conv_layers)
    parallelle_conv_layers:add(conv_layers:clone('weight', 'bias', 'gradWeight', 'gradBias', 'running_mean', 'running_std', 'running_var'))

    local net = nn.Sequential()
    --net:add(nn.SplitTable(2))
    net:add(conv_layers)

    --[[local mean = nn.Sequential()
    mean:add(nn.CAddTable())
    mean:add(nn.MulConstant(0.5))
    
    local subAndMean = nn.ConcatTable()
    subAndMean:add(nn.CSubTable())
    subAndMean:add(mean)
    net:add(subAndMean)

    net:add(nn.JoinTable(2))]]

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

    net:add(nn.Linear(last_layer_size, args.rnd_out))

    print(net)

    return net

end
