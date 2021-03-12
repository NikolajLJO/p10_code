--[[
Copyright (c) 2014 Google Inc.

See LICENSE file for full terms of limited license.
]]

require "nn"
require "image"

local scale = torch.class('nn.Scale', 'nn.Module')


function scale:__init(height, width, useRGB, thicken_image)
    self.height = height
    self.width = width
    self.useRGB = useRGB
    self.thicken_image = thicken_image
end


function scale:forward(x)

    if self.thicken_image then

        local x_copy = x
        if x_copy:dim() > 3 then
            x_copy = x_copy[1]
        end

        local result_tmp
        local result = x_copy:clone()

        if self.useRGB then
            result = image.rgb2y(result)
        end

        result_tmp = image.translate(result, 0, 1)
        result = torch.cmax(result, result_tmp)

        result_tmp = image.translate(result, 1, 0)
        result = torch.cmax(result, result_tmp)

        result_tmp = image.translate(result, 1, 1)
        result = torch.cmax(result, result_tmp)

        result = image.scale(result, self.width, self.height, 'bilinear')

        return result

    else

        local x = x
        if x:dim() > 3 then
            x = x[1]
        end

        if self.useRGB then
            result = image.rgb2y(result)
        end

        x = image.scale(x, self.width, self.height, 'bilinear')

        return x
    end
end


function scale:updateOutput(input)
    return self:forward(input)
end


function scale:float()
end
