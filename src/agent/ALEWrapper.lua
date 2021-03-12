local aleWrapper = torch.class('dqn.ALEWrapper')

function aleWrapper:__init(args)

    self.game_name              = args.env

    -- Small hack to deal with the dirty terminal signal in the ALE (as explained in the paper)
    self.episode_start_buffer                          = {}
    self.episode_start_buffer["montezuma_revenge"]     = 55

    self.episode_start_buffer   = self.episode_start_buffer[self.game_name] or 0

    self.useRGB                 = args.env_params.useRGB
    self.terminal_on_life_loss  = args.env_params.terminal_on_life_loss
    self.actrep                 = args.actrep
    self.random_starts          = args.random_starts

    self.use_SDL = false

    self.ffi = require 'ffi'

    self.ffi.cdef("typedef struct ALEInterface ALEInterface;")
    self.ffi.cdef("typedef int Action;")
    self.ffi.cdef("typedef int reward_t;")

    self.ffi.cdef(self:readContent(paths.thisfile("alewrap.inl")))

    self.lib = self.ffi.load(paths.thisfile("libalewrap.so"))

    self.lib.setInt("random_seed", args.seed)

    self.lib.setFloat("repeat_action_probability", args.env_params.repeat_action_probability)

    self.lib.setBool("color_averaging", true)

    if self.use_SDL then
        self.lib.setBool("display_screen", true)
        self.lib.setBool("sound", true)
    end

    self.action_set = nil

    self:loadGame(args.game_path .. self.game_name .. ".bin")

    self.state = {}
    self.state.lives = nil
    self.state.all_lives_lost = false
    self.state.screen = nil
    self.state.reward = 0
    self.state.terminal = false
end


function aleWrapper:loadGame(rom_file)

    self.lib.loadROM(rom_file)

    local nactions = self.lib.numLegalActions()
    local actions = torch.IntTensor(nactions)
    self.lib.legalActions(torch.data(actions), actions:nElement())

    self.action_set = actions:storage():totable()
end


function aleWrapper:getLegalActions()

    return self.action_set
end


function aleWrapper:getState()

    if not self.state.screen then
        self:updateScreen()
    end

    return self.state.screen:float():div(255), self.state.reward, self.state.terminal, self.state.all_lives_lost
end


function aleWrapper:stepSingleFrame(action)

    -- Apply the action and get the resulting reward
    local lives = self.lib.get_lives()
    local reward = self.lib.act(action)
    local terminal = self.lib.game_over()

    self.state.all_lives_lost = terminal

    if self.terminal_on_life_loss and self.state.lives and (lives < self.state.lives) then
        terminal = true
    end

    self.state.lives = lives

    return reward, terminal
end


-- Function plays one random action in the game and return game state.
function aleWrapper:randomStep()

    return self:stepSingleFrame(self.action_set[torch.random(#self.action_set)])
end


function aleWrapper:step(action)

    -- accumulate rewards over actrep action repeats
    local cumulated_reward = 0
    local frame, reward, terminal, lives

    for i = 1, self.actrep do

        -- Take selected action; ATARI games' actions start with action "0".
        reward, terminal = self:stepSingleFrame(action)

        -- accumulate instantaneous reward
        cumulated_reward = cumulated_reward + reward

        -- Loosing a life will trigger a terminal signal in training mode.
        -- We assume that a "life" IS an episode during training, but not during testing
        --[[
        if training and lives and lives < self._state.lives then
            terminal = true
        end
        --]]

        -- game over, no point to repeat current action
        if terminal then
            break
        end
    end

    self:updateScreen()

    self.state.reward = cumulated_reward
    self.state.terminal = terminal

    return self:getState()
end


function aleWrapper:updateScreen()

    if not self.state.screen then
        if self.useRGB then
            self.state.screen = torch.ByteTensor(3, 210, 160)
        else
            self.state.screen = torch.ByteTensor(210, 160)
        end
    end

    if self.useRGB then
        self.lib.fillRgbFromPalette(torch.data(self.state.screen), self.state.screen:nElement())
    else
        self.lib.fillGrayscaleFromPalette(torch.data(self.state.screen), self.state.screen:nElement())
    end
end


function aleWrapper:newGame()

    if self.state.all_lives_lost then

        self.lib.reset_game()

        self.state.lives = nil
        self.state.all_lives_lost = false
        self.state.screen = nil
        self.state.reward = 0
        self.state.terminal = false

    else

        for i = 1, self.episode_start_buffer + 1 do
            self:stepSingleFrame(0)
        end

        self.state.reward = 0
        self.state.terminal = false
        self:updateScreen()

    end

    return self:getState()
end


--[[ Function advances the emulator state until a new (random) game starts and
returns this state.
]]
function aleWrapper:nextRandomGame(k)

    k = k or torch.random(self.random_starts)

    self:newGame()

    local terminal

    for i = 1, k - 1 do

        _, terminal = self:stepSingleFrame(0)

        if terminal then
            print(string.format('WARNING: Terminal signal received after %d 0-steps', i))
        end
    end

    return self:getState()
end


-- Reads the whole content of the specified file.
function aleWrapper:readContent(path)

    local file = io.open(path)
    local content = file:read("*a")
    file:close()
    return content
end
