--[[
Copyright (c) 2014 Google Inc.

See LICENSE file for full terms of limited license.
]]

require "image"
require "Scale"

local function create_network(args)

    return nn.Scale(84, 84, args.useRGB, args.thicken_image)
end

return create_network
