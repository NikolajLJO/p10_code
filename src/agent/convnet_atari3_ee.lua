--[[
Copyright (c) 2014 Google Inc.

See LICENSE file for full terms of limited license.
]]

require 'convnet_ee'

return function(args)
    args.n_units        = {16, 16, 16} --{32, 64, 64} -- {16, 32, 32} -- {32, 64, 64} -- {8, 16, 16} --{16, 32, 32} -- {32, 64, 64}
    args.filter_size    = {8, 4, 3}
    args.filter_stride  = {4, 2, 1}
    args.n_hid          = {128} --{512} -- {512} -- {128}
    args.nl             = nn.Rectifier

    return create_network(args)
end

