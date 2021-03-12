require 'nn'

do

    local CustomLinear, parent = torch.class('nn.CustomLinear', 'nn.Linear')

    -- override the constructor to have the additional range of initialization
    function CustomLinear:__init(inputSize, outputSize, mean, std)
        parent.__init(self,inputSize, outputSize)
        self:reset(mean, std)
    end

    -- override the :reset method to use custom weight initialization.
    function CustomLinear:reset(mean, stdv)

        if mean and stdv then

            -- adapted from https://discuss.pytorch.org/t/implementing-truncated-normal-initializer/4778/5

            ---[[
            u1 = self.weight:clone()
            u1 = u1:rand(u1:size()) * (1 - math.exp(-2)) + math.exp(-2)

            u2 = self.weight:clone()
            u2 = u2:rand(u2:size())

            z = torch.sqrt(-2 * u1:clone():log()):cmul((2 * math.pi * u2):cos())

            self.weight:copy(z * stdv + mean)

            --[[
            u1_bias = self.bias:clone()
            u1_bias = u1_bias:rand(u1_bias:size()) * (1 - math.exp(-2)) + math.exp(-2)

            u2_bias = self.bias:clone()
            u2_bias = u2_bias:rand(u2_bias:size())

            z_bias = torch.sqrt(-2 * u1_bias:clone():log()):cmul((2 * math.pi * u2_bias):cos())

            self.bias:copy(z_bias * stdv + mean)
            --]]

            self.bias:zero()

            --self.weight:normal(mean, stdv)
            --self.bias:normal(mean, stdv)
            --self.bias:fill(0.1)
            --self.weight:uniform(-math.sqrt(3) * stdv, math.sqrt(3) * stdv)
            --self.bias:zero()
            --self.weight:normal(mean, stdv)
        else
            self.weight:normal(0, 1)
            self.bias:normal(0, 1)
            --self.bias:fill(0.1)
            --self.bias:zero()
        end
    end

end
