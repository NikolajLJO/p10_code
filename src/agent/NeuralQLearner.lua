--[[
Copyright (c) 2014 Google Inc.

See LICENSE file for full terms of limited license.
]]

if not dqn then
    require 'initenv'
end

local nql = torch.class('dqn.NeuralQLearner')

function nql:__init(args)

    -- Set up the log file directory
    self.logfilePath = args.logfile_dir .. os.date("%Y%m%d.%H%M%S") .. "/"
    self.logfile     = self.logfilePath .. "log.txt"

    os.execute("mkdir " .. self.logfilePath)

    self.game_episode_end_buffer                            = {}
    self.game_episode_end_buffer["freeway"]                 = 0
    self.game_episode_end_buffer["frostbite"]               = 30
    self.game_episode_end_buffer["montezuma_revenge"]       = 30
    self.game_episode_end_buffer["qbert"]                   = 30
    self.game_episode_end_buffer["venture"]                 = 80
    self.game_episode_end_buffer["ms_pacman"]               = 0
    self.game_episode_end_buffer["private_eye"]             = 0

    self.game_name                                          = args.game_name
    self.episode_end_buffer                                 = self.game_episode_end_buffer[self.game_name] or 0

    self:log("Game name: " .. self.game_name)
    self:log("Episode end buffer: " .. self.episode_end_buffer)

    self.useRGB                              = args.useRGB
    self.thicken_image                       = args.thicken_image

    self.episode_count                       = 0

    -- For debugging
    self.dump_counter                        = 0
    
    self.node_distances                      = {}
    self.bg_visited                          = false
    self.first_pass                          = true

    self.nodes                               = {}
    self.max_nodes                           = args.max_nodes
    self.nodes_visited                       = {}
    self.frames_since_reward                 = 0
    self.frames_since_reward_exceeded        = false

    self.first_screen                        = nil
    self.pending_node_info                   = nil
    self.best_pending_node_info              = nil

    self.node_addition_time_gap              = args.partition_add_initial_time_gap
    self.partition_add_time_mult             = args.partition_add_time_mult
    self.partition_visit_rate_mom            = args.partition_visit_rate_mom
    self.time_since_node_addition            = 0
    self.max_pellet_reward                   = args.max_pellet_reward

    self.current_node_num                    = -1

    self.win_main                            = nil
    self.distance_debug_window               = dqn.DebugWindow{width=600, height=350, caption="distance debug"}

    self.current_episode                     = {}
    self.loaded_episode                      = {}

    self.dump_convolutional_filters          = args.dump_convolutional_filters

    self.state_dim   = args.state_dim -- State dimensionality.
    self.actions     = args.actions
    self.n_actions   = #self.actions
    self.verbose     = args.verbose
    self.best        = args.best

    --- epsilon annealing
    self.ep_start    = args.ep or 1
    self.ep          = self.ep_start -- Exploration probability.
    self.ep_end      = args.ep_end or self.ep
    self.ep_endt     = args.ep_endt or 1000000

    ---- learning rate annealing
    self.lr_start                   = args.lr or 0.01 --Learning rate.
    self.lr                         = self.lr_start
    self.lr_end                     = args.lr_end or self.lr
    self.lr_endt                    = args.lr_endt or 1000000

    self.lr_start_image_compare     = args.lr_image_compare or 0.01
    self.lr_image_compare           = self.lr_start_image_compare
    self.lr_end_image_compare       = args.lr_end_image_compare or self.lr_image_compare
    self.lr_endt_image_compare      = args.lr_endt_image_compare or 1000000

    self.mom_image_compare          = args.mom_image_compare

    self.wc                     = args.wc or 0  -- L2 weight cost.
    self.reg_lambda             = args.reg_lambda or 0  -- L1 weight cost.
    self.reg_alpha              = args.reg_alpha or 0  -- L2 weight cost.
    self.minibatch_size         = args.minibatch_size or 1
    self.valid_size             = args.valid_size or 500

    --- Learning parameters
    self.discount                  = args.discount or 0.99 --Discount factor.
    self.update_freq               = args.update_freq or 1
    self.image_compare_update_freq = args.image_compare_update_freq or 1

    -- Number of points to replay per learning step.
    self.n_replay                  = args.n_replay or 1
    self.ee_n_replay               = args.ee_n_replay or 1
    self.RND_n_replay              = args.RND_n_replay or 1

    self.q_learn_min_partitions             = args.q_learn_min_partitions
    self.double_dqn                         = args.double_dqn
    self.exploration_style                  = args.exploration_style
    self.exploration_schedule_style         = args.exploration_schedule_style
    self.q_learn_MC_proportion              = args.q_learn_MC_proportion
    self.non_reward_frames_before_full_eps  = args.non_reward_frames_before_full_eps

    -- Number of steps after which learning starts.
    self.learn_start                               = args.learn_start or 0
    
    -- RND variables
    self.RND_learn_start                           = args.RND_learn_start
    self.RND_learn_end                             = args.RND_learn_end

    -- Variables related to image compare and node creation / selection
    self.ee_learn_start                            = args.ee_learn_start
    self.ee_learn_end                              = args.ee_learn_end
    self.step_count_when_learning_began            = -1
    self.partition_creation_start                  = args.partition_creation_start
    self.image_compare_node_creation_begun         = false

    self.partition_update_freq                     = args.partition_update_freq
    self.node_change_counter                       = 0
    self.do_time_flip                              = args.do_time_flip
    self.logger_steps                              = args.logger_steps
    self.debug_state_change_rate                   = args.debug_state_change_rate
    self.debug_state_change_steps                  = self.debug_state_change_rate
    self.reward_scale                              = args.reward_scale
    self.ee_comparison_policy                      = args.ee_comparison_policy
    self.ee_time_sep_constant_m                    = args.ee_time_sep_constant_m
    self.ee_discount                               = args.ee_discount
    self.ee_MC_proportion                          = args.ee_MC_proportion
    self.ee_MC_clip                                = args.ee_MC_clip
    self.training_method                           = args.training_method
    self.adam_epsilon                              = args.adam_epsilon
    self.rms_prop_epsilon                          = args.rms_prop_epsilon
    self.evaluation_mode_required                  = args.evaluation_mode_required
    self.partition_selection_style                 = args.partition_selection_style
    self.stored_displacements_refresh_steps        = args.stored_displacements_refresh_steps
    self.show_display                              = false
    self.display_image                             = nil
    self.print_action_probs                        = false
    self.pseudocount_rewards_on                    = args.pseudocount_rewards_on
    self.ee_beta                                   = args.ee_beta
    self.current_origin_state                      = nil
    self.current_debug_state                       = nil
    self.current_debug_caption                     = "Random valid frame from cache"

    -- For error stats
    self.stats_averaging_n                         = 10000
    self.err_arr_train                             = {}
    self.err_arr_test                              = {}
    self.err_arr_train_idx                         = 1
    self.err_arr_test_idx                          = 1

    self.episode_score                             = 0
    self.episode_scores                            = {}
    self.episode_scores_idx                        = 1
    self.episode_scores_averaging_n                = 1000


    -- Counters
    self.first_node_set_timer                      = 0

     -- Size of the transition table.
    self.replay_memory                             = args.replay_memory or 1000000

    self.hist_len                       = args.hist_len or 1
    self.ee_histLen                     = args.ee_histLen or 1
    self.RND_histLen                    = args.RND_histLen or 1
    self.rescale_r                      = args.rescale_r
    self.max_reward                     = args.max_reward
    self.min_reward                     = args.min_reward
    self.clip_delta                     = args.clip_delta
    self.target_q                       = args.target_q
    self.bestq                          = 0

    self.gpu                            = args.gpu

    self.ncols                          = args.ncols or 1  -- number of color channels in input
    self.input_dims                     = args.input_dims or {self.hist_len * self.ncols, 84, 84}
    self.preproc                        = args.preproc  -- name of preprocessing network
    self.histType                       = args.histType or "linear"  -- history type to use
    self.histSpacing                    = args.histSpacing or 1
    self.ee_histSpacing                 = args.ee_histSpacing or 1
    self.nonTermProb                    = args.nonTermProb or 1
    self.bufferSize                     = args.bufferSize or 512

    self.transition_params              = args.transition_params or {}

    self.network                        = args.network
    self.image_compare_network          = args.image_compare_network

    self.image_dump_counter = 1

    -- For debugging
    self.action_names = {}
    self.action_names[1] = "NOOP"
    self.action_names[2] = "FIRE"
    self.action_names[3] = "UP"
    self.action_names[4] = "RIGHT"
    self.action_names[5] = "LEFT"
    self.action_names[6] = "DOWN"
    self.action_names[7] = "UPRIGHT"
    self.action_names[8] = "UPLEFT"
    self.action_names[9] = "DOWNRIGHT"
    self.action_names[10] = "DOWNLEFT"
    self.action_names[11] = "UPFIRE"
    self.action_names[12] = "RIGHTFIRE"
    self.action_names[13] = "LEFTFIRE"
    self.action_names[14] = "DOWNFIRE"
    self.action_names[15] = "UPRIGHTFIRE"
    self.action_names[16] = "UPLEFTFIRE"
    self.action_names[17] = "DOWNRIGHTFIRE"
    self.action_names[18] = "DOWNLEFTFIRE"

    self.zero_aux_rewards = {}

    for i = 1, self.n_actions do
        self.zero_aux_rewards[i] = 0
    end

    -- Print args to log file in alphabetical order
    self:log("--------------------------------------------------")

    local alphabetical_keys = {}
    for n in pairs(args) do
        table.insert(alphabetical_keys, n)
    end
    table.sort(alphabetical_keys)

    for k, v in ipairs(alphabetical_keys) do
        self:log(tostring(v) .. " = " .. tostring(args[v]))
    end

    self:log("--------------------------------------------------")
    self:log("")

    -- check whether there is a network file
    local network_function
    if not (type(self.network) == 'string') then
        error("The type of the network provided in NeuralQLearner" ..
              " is not a string!")
    end


--#####################################
--###########   Q Network   ###########
--#####################################

    local msg, err = pcall(require, self.network)
    if not msg then
        -- try to load saved agent
        self.network = torch.load(args.network)
    else
        print('Creating Agent Network from ' .. self.network)
        self.network = err
        self.network = self:network()
    end


--########################################
--###########   RDN networks   ###########
--########################################
    msg, err = pcall(require, "convnet_atari3_PRND")
    if not msg then
        -- try to load saved agent
        print('Loading Image Compare Network from ' .. "convnet_atari3_PRND")
        self:log('Loading Image Compare Network from ' .. "convnet_atari3_PRND")
        self.RND_P_network = torch.load("convnet_atari3_PRND")
    else
        print('Creating Image Compare Network from ' .. "convnet_atari3_PRND")
        self.RND_P_network = err
        self.RND_P_network = self:RND_P_network()
    end
    self:log(tostring(self.RND_P_network))
    self:log("--------------------------------------------------")
    self:log("")
    
    msg, err = pcall(require, "convnet_atari3_TRND")
    if not msg then
        -- try to load saved agent
        print('Loading Image Compare Network from ' .. "convnet_atari3_TRND")
        self:log('Loading Image Compare Network from ' .. "convnet_atari3_TRND")
        self.RND_T_network = torch.load("convnet_atari3_TRND")
    else
        print('Creating Image Compare Network from ' .. "convnet_atari3_TRND")
        self.RND_T_network = err
        self.RND_T_network = self:RND_T_network()
    end
    self:log(tostring(self.RND_T_network))
    self:log("--------------------------------------------------")
    self:log("")


    -- Finalisation
    if self.gpu and self.gpu >= 0 then
        self.network:cuda()
        self.RND_P_network:cuda()
        self.RND_T_network:cuda()
    else
        self.network:float()
        self.RND_P_network:float()
        self.RND_T_network:float()
    end

    -- Load preprocessing network.
    if not (type(self.preproc == 'string')) then
        error('The preprocessing is not a string')
    end
    msg, err = pcall(require, self.preproc)
    if not msg then
        error("Error loading preprocessing net")
    end
    self.preproc = err
    self.preproc = self:preproc()
    self.preproc:float()

    if self.gpu and self.gpu >= 0 then
        self.network:cuda()
        self.RND_P_network:cuda()
        self.RND_T_network:cuda()
        self.tensor_type = torch.CudaTensor
    else
        self.network:float()
        self.RND_P_network:float()
        self.RND_T_network:float()
        self.tensor_type = torch.FloatTensor
    end

    -- Create transition table.
    ---- assuming the transition table always gets floating point input
    ---- (Foat or Cuda tensors) and always returns one of the two, as required
    ---- internally it always uses ByteTensors for states, scaling and
    ---- converting accordingly

    local transition_args = {
        agent = self,
        stateDim = self.state_dim,
        numActions = self.n_actions,
        histLen = self.hist_len,
        gpu = self.gpu,
        maxSize = self.replay_memory,
        histType = self.histType,
        histSpacing = self.histSpacing,
        nonTermProb = self.nonTermProb,
        bufferSize = self.bufferSize,
        episode_end_buffer = self.episode_end_buffer,
        discount = self.discount,
        ee_discount = self.ee_discount
    }
    self.transitions = dqn.TransitionTable(transition_args)

    self.numSteps = 0 -- Number of perceived states.

    self.lastState              = nil
    self.lastAction             = nil
    self.lastActionGreedy       = false
    self.lastActionProbs        = nil
    self.last_node_num          = -1
    self.last_nodes_visited     = {}

    self.v_avg = 0 -- V running average.
    self.tderr_avg = 0 -- TD error running average.

    self.q_max = 1
    self.r_max = 1

    self.w, self.dw = self.network:getParameters()
    self.RND_w, self.RND_dw = self.RND_P_network:getParameters()

    self.dw:zero()
    self.RND_dw:zero()

    self.deltas = self.dw:clone():fill(0)

    self.tmp = self.dw:clone():fill(0)
    self.g = self.dw:clone():fill(0)
    self.g2 = self.dw:clone():fill(0)

    if self.target_q then
        self.target_network = self.network:clone()
    end

    self.config_adam = {}
    self.config_adam.learningRate = self.lr_image_compare --0.0001
    --self.config_adam.learningRateDecay = ?
    self.config_adam.weightDecay = (1.0 - self.reg_alpha) * self.reg_lambda
    --self.config_adam.beta1 = ?
    --self.config_adam.beta2 = ?
    self.config_adam.epsilon = self.adam_epsilon

    self.config_adam_q = {}
    self.config_adam_q.learningRate = self.lr_image_compare --0.0001
    --self.config_adam_q.learningRateDecay = ?
    self.config_adam_q.weightDecay = (1.0 - self.reg_alpha) * self.reg_lambda
    --self.config_adam_q.beta1 = ?
    --self.config_adam_q.beta2 = ?
    self.config_adam_q.epsilon = self.adam_epsilon --0.0003125

    self.config_sgd = {}
    self.config_sgd.learningRate = self.lr_image_compare --0.0001
    self.config_sgd.momentum = self.mom_image_compare
    self.config_sgd.dampening = 0
    self.config_sgd.weightDecay = (1.0 - self.reg_alpha) * self.reg_lambda
    self.config_sgd.nesterov = true

    self.config_rms_prop = {}
    self.config_rms_prop.learningRate = self.lr_image_compare --0.0001
    -- I think the name 'alpha' is confusing, should it be 'momentum'?
    -- In other RMS prop descriptions, alpha is the learning rate, i.e. it's not the same thing as here
    self.config_rms_prop.alpha = self.mom_image_compare
    self.config_rms_prop.epsilon = self.rms_prop_epsilon
    self.config_rms_prop.weightDecay = (1.0 - self.reg_alpha) * self.reg_lambda
end


function nql:log(logStr)
    file = io.open(self.logfile, "a")
    file:write(logStr .. "\n")
    file:close()
end


function nql:reset(state)
    if not state then
        return
    end
    self.best_network = state.best_network
    self.network = state.model
    self.w, self.dw = self.network:getParameters()
    self.dw:zero()
    self.numSteps = 0
    print("RESET STATE SUCCESFULLY")
end


function nql:preprocess(rawstate)
    if self.preproc then
        -- After the call to reshape, the return value is no longer a "displayable" image
        -- It just becomes one long column vector containing the state
        return self.preproc:forward(rawstate:float())
                    :clone():reshape(self.state_dim)
    end

    return rawstate
end


function nql:getQUpdate(args)

    local s, a, r, s2, term, ret_mc
    local q, q2, q_all, q2_max

    s = args.s
    a = args.a
    r = args.r
    s2 = args.s2
    term = args.term
    ret_mc = args.ret_mc

    -- The order of calls to forward is a bit odd in order
    -- to avoid unnecessary calls (we only need 2).

    -- delta = r + (1-terminal) * gamma * max_a Q(s2, a) - Q(s, a)
    term = term:clone():float():mul(-1):add(1)

    local target_q_net
    if self.target_q then
        target_q_net = self.target_network
    else
        target_q_net = self.network
    end

    if self.double_dqn then

        local q2_all = target_q_net:forward(s2):float()
        local q2_all_live_net = self.network:forward(s2):float()

        -- The best action is determined by the "live" network
        -- However, it's then evaluated by the target network
        q_all = self.network:forward(s):float()

        local max_val_live_net = {}
        q2_max = torch.FloatTensor(self.minibatch_size)

        for i = 1, self.minibatch_size do

            max_val_live_net[i] = q2_all_live_net[i][1]
            q2_max[i] = q2_all[i][1]

            for j = 2, self.n_actions do

                if q2_all_live_net[i][j] > max_val_live_net[i] then
                    max_val_live_net[i] = q2_all_live_net[i][j]
                    q2_max[i] = q2_all[i][j]
                end
            end
        end

    else

        q2_max = target_q_net:forward(s2):float():max(2)
        q_all = self.network:forward(s):float()
    end

    -- Compute q2 = (1-terminal) * gamma * max_a Q(s2, a)
    q2 = q2_max:clone():mul(self.discount):cmul(term):float()


    local r_adj
    local curr_val
    local delta = torch.FloatTensor(r:size(1))
    local delta_bootstrap
    local delta_mc

    for i = 1, r:size(1) do -- ##### TO DO: Might be more efficient if the replay memory just returned the rewards as a table #####

        if self.rescale_r then
            r_adj = r[i] / self.r_max
        else
            r_adj = r[i]
        end

        curr_val = q_all[i][a[i]]

        delta_bootstrap = r_adj + q2[i] - curr_val
        delta_mc = ret_mc[i] - curr_val

        delta[i] = (1 - self.q_learn_MC_proportion) * delta_bootstrap + self.q_learn_MC_proportion * delta_mc
    end


    if self.clip_delta then
        delta[delta:ge(self.clip_delta)] = self.clip_delta
        delta[delta:le(-self.clip_delta)] = -self.clip_delta
    end

    local targets = torch.zeros(self.minibatch_size, self.n_actions):float()
    for i = 1, math.min(self.minibatch_size, a:size(1)) do
        targets[i][a[i]] = delta[i]
    end

    if self.gpu >= 0 then
        targets = targets:cuda()
    end

    return targets, delta, q2_max
end


function nql:qLearnMinibatch()

    -- Perform a minibatch Q-learning update:
    -- w += alpha * (r + gamma max Q(s2,a2) - Q(s,a)) * dQ(s,a)/dw
    assert(self.transitions:size() > self.minibatch_size)

    local s, a, r, s2, term, ret_mc, node_visits_s, node_visits_s2 = self.transitions:sample(self.minibatch_size)

    s:reshape(s:size(1), 4 * 84 * 84)
    s2:reshape(s2:size(1), 4 * 84 * 84)

    if self.gpu >= 0 then
        node_visits_s = node_visits_s:cuda()
        node_visits_s2 = node_visits_s2:cuda()
    end

    s = torch.cat(s, node_visits_s, 2)
    s2 = torch.cat(s2, node_visits_s2, 2)

    local targets, delta, q2_max = self:getQUpdate{s=s, a=a, r=r, s2=s2, term=term, ret_mc=ret_mc}

    -- zero gradients of parameters
    self.dw:zero()

    -- get new gradient
    self.network:backward(s, targets)

    -- add weight cost to gradient
    self.dw:add(-self.wc, self.w)

    -- compute linearly annealed learning rate
    local t = math.max(0, self.numSteps - self.learn_start)
    self.lr = (self.lr_start - self.lr_end) * (self.lr_endt - t)/self.lr_endt +
                self.lr_end
    self.lr = math.max(self.lr, self.lr_end)



    ---[[
    self.config_adam_q.learningRate = 0.0000625
    --self.config_adam_q.learningRate = 0.00025

    function feval(params)

        local loss = 0 --dummy

        return loss, -self.dw
    end

    optim.adam(feval, self.w, self.config_adam_q)
    --]]



    --[[
    -- use gradients
    self.lr = 0.00025
    local mom = 0 --0.9
    local decay = 0.95
    local eps = 0.0001

    -- See https://arxiv.org/pdf/1308.0850.pdf, page 23

    -- g = g * 0.95 + 0.05 * dw (this is the g_i line)
    self.g:mul(decay):add(1 - decay, self.dw)

    -- tmp = dw.^2
    self.tmp:cmul(self.dw, self.dw)

    -- g2 = g2 * 0.95 + 0.05 * tmp (this, combined with the above, is the n_i line)
    self.g2:mul(decay):add(1 - decay, self.tmp)

    -- tmp = g^2
    self.tmp:cmul(self.g, self.g)

    -- tmp = sqrt(g2 - tmp + 0.01) (so, tmp = sqrt(n_i - g_i^2 + 0.01), i.e. the denominator)
    self.tmp:mul(-1)
    self.tmp:add(self.g2)
    self.tmp:add(eps)
    self.tmp:sqrt()

    -- accumulate update
    self.deltas:mul(mom):addcdiv(self.lr, self.dw, self.tmp)
    self.w:add(self.deltas)
    --]]
end


function nql:getAverageEpisodeScore()

    local sumScores = 0
    for i = 1, #self.episode_scores do
        sumScores = sumScores + self.episode_scores[i]
    end

    return sumScores / #self.episode_scores
end


function nql:getSumSqErr()

    local sum_err_train = 0
    for i = 1, #self.err_arr_train do
        sum_err_train = sum_err_train + self.err_arr_train[i]
    end

    local sum_err_test = 0
    for i = 1, #self.err_arr_test do
        sum_err_test = sum_err_test + self.err_arr_test[i]
    end

    return
        sum_err_train / #self.err_arr_train,
        sum_err_test / #self.err_arr_test

end


function nql:get_image_compare_update(args)

    local net_input_for_boot = args.net_input_for_boot
    local s = args.s
    local a = args.a
    local a_count = args.a_count
    local flip = args.flip
    local action_probs = args.action_probs

    self.target_image_compare_network:evaluate()
    local pred_boot = self.target_image_compare_network:forward(net_input_for_boot):float()

    self.image_compare_network:training()
    local pred = self.image_compare_network:forward(s):float()

    local labels_tab = {}
    local labels_mc_tab = {}

    for i = 1, self.minibatch_size do

        labels_tab[i] = {}
        labels_mc_tab[i] = {}

        for j = 1, self.n_actions do
            labels_tab[i][j] = 0
            labels_mc_tab[i][j] = 0
        end
    end

    for i = 1, self.minibatch_size do

        local mc_proportion = self.ee_MC_proportion

        for j = 1, self.n_actions do

            local reward

            if self.ee_comparison_policy == "uniform" then

                if a[i] == j then
                    reward = 1 - (1 / self.n_actions)
                else
                    reward = -1 / self.n_actions
                end

            elseif self.ee_comparison_policy == "current" then

                if a[i] == j then
                    reward = 1 - action_probs[i][j]
                else
                    reward = -action_probs[i][j]
                end

            else
                error("Unknown comparison policy!")

            end

            reward = reward * self.reward_scale

            local bootstrapped_val = reward + self.ee_discount * pred_boot[i][j]
            labels_mc_tab[i][j] = a_count[i][j]

            if flip then
                bootstrapped_val = bootstrapped_val * -1
                labels_mc_tab[i][j] = labels_mc_tab[i][j] * -1
            end

            local delta_mc = labels_mc_tab[i][j] - pred[i][j]
            delta_mc = math.max(delta_mc, -self.ee_MC_clip)
            delta_mc = math.min(delta_mc, self.ee_MC_clip)

            local delta_bootstrap = bootstrapped_val - pred[i][j]

            local combined_delta = mc_proportion * delta_mc + (1 - mc_proportion) * delta_bootstrap

            labels_tab[i][j] = pred[i][j] + combined_delta
        end
    end

    local labels = torch.FloatTensor(self.minibatch_size, self.n_actions)

    for i = 1, self.minibatch_size do
        for j = 1, self.n_actions do
            labels[i][j] = labels_tab[i][j]
        end
    end

    return pred, labels
end


function nql:image_compare_learn_minibatch(args)

    local flip = false
    if self.do_time_flip and torch.uniform() < 0.5 then
        flip = true
    end

    local replay_memory = args.replay_memory

    -- Perform a minibatch update of the image comparison network
    local s, s2, smid, a, a_count, action_probs = replay_memory:sample_ic{batch_size=self.minibatch_size}

    s:resize(self.minibatch_size, 1, 84, 84)
    s2:resize(self.minibatch_size, 1, 84, 84)
    smid:resize(self.minibatch_size, 1, 84, 84)

    -- For debugging
    --[[
    local img = s[1]:clone()
    image.save(self.logfilePath .. "dummy1.png", img)

    img = s2[1]:clone()
    image.save(self.logfilePath .. "dummy2.png", img)
    --]]

    -- compute linearly annealed learning rate
    local t = math.max(0, self.numSteps - self.step_count_when_learning_began)
    self.lr_image_compare = (self.lr_start_image_compare - self.lr_end_image_compare) * (self.lr_endt_image_compare - t) / self.lr_endt_image_compare + self.lr_end_image_compare
    self.lr_image_compare = math.max(self.lr_image_compare, self.lr_end_image_compare)

    self.config_adam.learningRate = self.lr_image_compare
    self.config_adam.momentum = self.mom_image_compare

    self.config_sgd.learningRate = self.lr_image_compare
    self.config_sgd.momentum = self.mom_image_compare

    self.config_rms_prop.learningRate = self.lr_image_compare
    self.config_rms_prop.alpha = self.mom_image_compare

    local net = self.image_compare_network

    local net_input
    if not flip then
        net_input = self:merge_distance_net_states{s1=s, s2=s2}
    else
        net_input = self:merge_distance_net_states{s1=s2, s2=s}
    end

    local net_input_for_boot = self:merge_distance_net_states{s1=smid, s2=s2}

    local w = self.image_compare_w
    local dw = self.image_compare_dw
    local g = self.image_compare_g
    local g2 = self.image_compare_g2
    local tmp = self.image_compare_tmp
    local deltas = self.image_compare_deltas
    local learning_rate = self.lr_image_compare
    local momentum = self.mom_image_compare

    assert(replay_memory:size() > self.minibatch_size)

    local pred_eval
    if self.evaluation_mode_required then
        self.image_compare_network:evaluate()
        pred_eval = self.image_compare_network:forward(net_input):clone():float()
        self.image_compare_network:training()
    end

    -- get new gradient
    local pred, labels = self:get_image_compare_update{net_input_for_boot=net_input_for_boot, s=net_input, a=a, a_count=a_count, flip=flip, action_probs=action_probs}

    if not self.evaluation_mode_required then
        pred_eval = pred:clone()
    end

    local criterion = nn.MSECriterion()

    local loss_eval = criterion:forward(pred_eval, labels)

    self.err_arr_train[self.err_arr_train_idx] = loss_eval * (1 / self.reward_scale)^2
    self.err_arr_train_idx = (self.err_arr_train_idx % self.stats_averaging_n) + 1

    function feval(params)

        dw:zero()

        local loss = criterion:forward(pred, labels)

        local dloss_doutputs = criterion:backward(pred, labels)
        if self.gpu >= 0 then
            dloss_doutputs = dloss_doutputs:cuda()
        end

        net:backward(net_input, dloss_doutputs)

        -- Gradient clipping
        --local maxGradient = 1
        --dw[dw:ge(maxGradient)] = maxGradient
        --dw[dw:le(-maxGradient)] = -maxGradient

        return loss, dw
    end

    if self.training_method == "adam" then

        optim.adam(feval, w, self.config_adam)

    elseif self.training_method == "sgd" then

        optim.sgd(feval, w, self.config_sgd)

    elseif self.training_method == "rmsprop" then

        optim.rmsprop(feval, w, self.config_rms_prop)

    elseif self.training_method == "rmsprop_dqn" then

        dw:zero()

        local dloss_doutputs = criterion:backward(pred, labels)
        net:backward(net_input, dloss_doutputs:cuda())

        -- regularisation
        local l1_adj = torch.sign(w):mul(-self.reg_lambda):mul(self.reg_alpha)
        local l2_adj = w:clone():mul(-self.reg_lambda):mul(1.0 - self.reg_alpha)
        local elastic_adj = l1_adj:add(l2_adj)

        dw:add(elastic_adj)

        -- use gradients
        -- Reason for negative sign on 'dw':
        -- See https://groups.google.com/forum/#!topic/deep-q-learning/_RFrmUALBQo
        g:mul(momentum):add(1.0 - momentum, -dw)
        tmp:cmul(-dw, -dw)
        g2:mul(momentum):add(1.0 - momentum, tmp)
        tmp:cmul(g, g)
        tmp:mul(-1)
        tmp:add(g2)
        tmp:add(0.01)
        tmp:sqrt()

        -- accumulate update
        deltas:mul(0):addcdiv(learning_rate, -dw, tmp)
        w:add(deltas)

    else

        error("Unknown training method!")
    end
end


function nql:sample_validation_data()
    local s, a, r, s2, term = self.transitions:sample(self.valid_size)
    self.valid_s    = s:clone()
    self.valid_a    = a:clone()
    self.valid_r    = r:clone()
    self.valid_s2   = s2:clone()
    self.valid_term = term:clone()
end


function nql:compute_validation_statistics()

    local targets, delta, q2_max = self:getQUpdate{s=self.valid_s, a=self.valid_a, r=self.valid_r, s2=self.valid_s2, term=self.valid_term}

    self.v_avg = self.q_max * q2_max:mean()
    self.tderr_avg = delta:clone():abs():mean()
end


function nql:merge_distance_net_states(args)

    if self.gpu and self.gpu >= 0 then
        return torch.cat(args.s1, args.s2, 2):cuda()
    else
        return torch.cat(args.s1, args.s2, 2):float()
    end
end



function nql:file_exists(filename)

    local f = io.open(filename, "r")
    if f ~= nil then
        io.close(f)
        return true
    else
        return false
    end
end


function nql:check_for_settings_files()

    -- check if we should quit (dodgy style because you can't check for keystrokes in Lua)
    if self.numSteps % 100 == 0 then

        local quitFile = self.logfilePath .. 'q'
        if self:file_exists(quitFile) then
            os.remove(quitFile)
            os.exit()
        end

        local toggleDisplayFile = self.logfilePath .. 'toggleDisplay'
        if self:file_exists(toggleDisplayFile) then
            os.remove(toggleDisplayFile)
            self.show_display = not self.show_display

            if not self.show_display then
                self.win_main = nil
            end
        end

        local toggleDistanceDebugFile = self.logfilePath .. 'toggleDistanceDebug'
        if self:file_exists(toggleDistanceDebugFile) then
            os.remove(toggleDistanceDebugFile)
            self.print_action_probs = not self.print_action_probs
        end
    end
end


--[[
function nql:get_self_directions(args)

    local s = (args.s):clone():reshape(1, 84, 84)
    local stochastic_inference = args.stochastic_inference or false

    local net_input = torch.repeatTensor(s, 2, 1, 1):reshape(1, 2, 84, 84)

    if self.gpu and self.gpu >= 0 then
        net_input = net_input:cuda()
    else
        net_input = net_input:float()
    end

    local net_output = self.image_compare_network:forward(net_input):float()

    local action_probs = {}

    for act_idx = 1, self.n_actions do
        action_probs[act_idx] = net_output[1][act_idx]
    end

    return action_probs
end
--]]


function nql:get_self_directions(args)

    local s = (args.s):clone():reshape(84, 84)

    local net_input = torch.FloatTensor(1, 2, 84, 84):fill(0)

    net_input[1][1]:copy(s)
    net_input[1][2]:copy(s)

    if self.gpu and self.gpu >= 0 then
        net_input = net_input:cuda()
    else
        net_input = net_input:float()
    end

    -- For debugging
    --[[
    local img = net_input[1][1]:clone():resize(84, 84)
    image.save(self.logfilePath .. "dummy1.png", img)

    img = net_input[1][2]:clone():resize(84, 84)
    image.save(self.logfilePath .. "dummy2.png", img)
    --]]

    local net_output = self:RND_calc_novelty(net_input)

    return net_output
end


function nql:get_novelty_to_and_from_state(args)

    local s = args.s

    local net_input = torch.FloatTensor(2 * #self.nodes, 2, 84, 84):fill(0)

    for i = 1, #self.nodes do
        net_input[i][1]:copy(self.nodes[i].s)
        net_input[i][2]:copy(s)
    end

    for i = 1, #self.nodes do
        local input_idx = i + #self.nodes
        net_input[input_idx][1]:copy(s)
        net_input[input_idx][2]:copy(self.nodes[i].s)
    end

    if self.gpu and self.gpu >= 0 then
        net_input = net_input:cuda()
    else
        net_input = net_input:float()
    end
    
    local net_output = {}
    for i=1, net_input:size(1) do
        net_output[i] = self:RND_calc_novelty(net_input[i])
    end
    local directions_to_current_state = {}
    for node_num = 1, #self.nodes do

        directions_to_current_state[node_num] = {}

        for i = 1, #net_output do
            directions_to_current_state[node_num] = net_output[node_num]
        end
    end

    local directions_from_current_state = {}
    for node_num = 1, #self.nodes do

        local output_idx = node_num + #self.nodes
        directions_from_current_state[node_num] = {}

        for i = 1, #net_output do
            directions_from_current_state[node_num] = net_output[output_idx]
        end
    end

    return novelty_to_current_state, novelty_from_current_state
end


function nql:update_nodes(args)

    local s = (args.s):clone():reshape(84, 84)
    --lars: her skal noeget ændes (lars has been here to RND the place up )
    local self_novelty = self:get_self_directions{s=s}

    if #self.nodes == 0 then

        if (#self.nodes < self.max_nodes) and not self.pending_node_info  then

            --local first_screen_reshaped = self.first_screen:clone():reshape(84, 84)

            local new_node = dqn.Node{s=s, idx=1, visit_pct_mom=self.partition_visit_rate_mom}

            self.pending_node_info = {}
            self.pending_node_info.node = new_node
            --self.pending_node_info.pending_time = self.episode_end_buffer + 1 -- Can add immediately
            self.pending_node_info.pending_time = 0
            --self.pending_node_info.rule_out_reason = {}
            self.pending_node_info.directions_to_current_state = {}
            --self.pending_node_info.self_directions = self:get_self_directions{s=first_screen_reshaped, stochastic_inference=false}
            self.pending_node_info.self_directions = self_nuvelty
            self.pending_node_info.episode_age = #self.current_episode
            self.pending_node_info.self_error = math.abs(self_novelty)
            self.pending_node_info.smallest_max_discrepancy = 0
            self.pending_node_info.closest_node = 0
            self.pending_node_info.closest_node_visited = 0
            self.pending_node_info.visited = 0
            self.pending_node_info.eval_time = 0
            self.pending_node_info.dir_to = {}
            self.pending_node_info.dir_from = {}

            self.nodes_visited[1] = true -- This will get wiped by "addEpisodeToExperienceCache" if a life is soon lost and the node isn't added
        end

        return
    end
    --[[
    --Temporarily add candidate to nodes for ease
    if self.best_pending_node_info then
        local new_node = self.best_pending_node_info.node

        self.nodes[new_node.idx] = new_node
    end --]]

    local directions_to_current_state, directions_from_current_state = self:get_novelty_to_and_from_state{s=s}

    --local overall_candidates = {}
    local max_discrepancies = {}
    --local sum_discrepancies = {}
    for i = 1, #self.nodes do
        --overall_candidates[i] = true
        max_discrepancies[i] = -math.huge
        --sum_discrepancies[i] = 0
    end

    --local rule_out_reason = {}

    -- Rule out nodes by comparing
    -- (A) Vector from existing node (i) to the current state
    -- vs
    -- (B) Vector from existing node (i) to existing node (j)
    -- Can't be (j) if the distance is too big.
    for node_num = 1, #self.nodes do
        --lars rnd kan finde denne udregning ( the calculation is changed)
        --local direction_diff = self.nodes[node_num]:get_direction_diff_to{directions_to_current_state=directions_to_current_state[node_num]}
        local novelty = self:RND_calc_novelty_between_two_states{from=self.nodes[node_num].s, to=s}
        
        --print("first direction_diff at node " .. node_num .. " = " .. (direction_diff[1] or -99))
        --print(" dir diff is so long: " .. #direction_diff)
        if novelty > max_discrepancies[node_num] then
            max_discrepancies[node_num] = novelty
        end
    end


    -- Rule out nodes by comparing
    -- (A) Vector from current state to existing node (j)
    -- vs
    -- (B) Vector from existing node (i) to existing node (j)
    -- Can't be (i) if the distance is too big.
    for node_num = 1, #self.nodes do
        --lars rnd kan finde denne udregning (been here done that)
        --local direction_diff = self.nodes[node_num]:get_direction_diff_to{directions_to_current_state=directions_to_current_state[node_num]}
        local novelty = self.RND_calc_novelty_between_two_states{from=s, to=self.nodes[node_num].s}
        
        --print("first direction_diff at node " .. node_num .. " = " .. (direction_diff[1] or -99))
        --print(" dir diff is so long: " .. #direction_diff)

        if novelty > max_discrepancies[node_num] then
            max_discrepancies[node_num] = novelty
        end
    end

    --[[
    --Remove the added candidate node
    if self.best_pending_node_info then
        table.remove(self.nodes)
    end--]]

    -- Update current node

    -- Version 1: Closest node favoured
    local smallest_max_discrepancy = math.huge
    for i = 1, #self.nodes do

        if max_discrepancies[i] < smallest_max_discrepancy then

            smallest_max_discrepancy = max_discrepancies[i]

            if self.partition_selection_style == "closest" then
                --and smallest_max_discrepancy <= self.image_compare_node_visit_thresh then
                --and overall_candidates[i] then

                self.current_node_num = i
            end
        end
    end

    -- Version 2: Oldest node favoured
    --local nodes_in_range = {}
    local current_node_set = false

    for i = 1, #self.nodes do

        --if overall_candidates[i] then

            --nodes_in_range[#nodes_in_range + 1] = i

            if not current_node_set
                and self.partition_selection_style == "oldest" then
                --and smallest_max_discrepancy <= self.image_compare_node_visit_thresh then
                --and overall_candidates[i] then

                self.current_node_num = i
                current_node_set = true
            end
        --end
    end
    
    table.insert(self.node_distances[self.current_node_num], smallest_max_discrepancy)
    
    --Add one to eval time
    if self.best_pending_node_info and self.first_pass then
        self.first_pass = false
        self.best_pending_node_info.eval_time = self.best_pending_node_info.eval_time + 1
    end
    
    --Check if cand subgoal would be visisted
    if self.best_pending_node_info and not self.bg_visited then
        local cand_max_disc = -math.huge
        for node_num = 1, #self.nodes do
            local direction_diff = self:get_subgoal_diff{s = s, pending_s = self.best_pending_node_info.s}

            for j = 1, #direction_diff do

                --if direction_diff[j] > self.image_compare_node_add_thresh then

                    -- If node node_num is still a candidate then remove it
                    --if overall_candidates[node_num] then
                        --overall_candidates[node_num] = false
                        --rule_out_reason[node_num] = {j, "from", directions_from_current_state[j], self.nodes[node_num].directions_to_node[j]}
                    --end
                --end

                --sum_discrepancies[node_num] = sum_discrepancies[node_num] + direction_diff[j]
                if direction_diff[j] > cand_max_disc then
                    cand_max_disc = direction_diff[j]
                end
            end
        end
        if cand_max_disc < smallest_max_discrepancy then
            self.bg_visited = true
            self.best_pending_node_info.visited = self.best_pending_node_info.visited + 1
        end  
    end
    
    --[[
    --Check if candidate would have been visited over current closest node
    if self.best_pending_node_info and not self.bg_visited then
        print("max_discrepancies[#max_discrepancies] = " .. max_discrepancies[#max_discrepancies])
        if max_discrepancies[#max_discrepancies] < smallest_max_discrepancy then
            self.bg_visited = true
            self.best_pending_node_info.visited = self.best_pending_node_info.visited + 1
        end        
    end--]]

    if self.current_node_num > 0 then

        if not self.nodes_visited[self.current_node_num] then
            self.frames_since_reward = 0
            if self.best_pending_node_info and self.current_node_num == self.best_pending_node_info.closest_node then
                 self.best_pending_node_info.closest_node_visited = self.best_pending_node_info.closest_node_visited + 1
            end
        end

        self.nodes_visited[self.current_node_num] = true
    end

    -- Add pending new node if necessary
    --[[
    if #nodes_in_range == 0
        and not self.pending_node_info
        and #self.nodes < self.max_nodes then
    --]]
    --print("smallest_max_discrepancy = " .. smallest_max_discrepancy)
    --if self.best_pending_node_info then
    --  print("self.best_pending_node_info.smallest_max_discrepancy = " .. self.best_pending_node_info.smallest_max_discrepancy)
    --end

    if ((not self.pending_node_info) and (not self.best_pending_node_info))
        or (self.pending_node_info and (smallest_max_discrepancy > self.pending_node_info.smallest_max_discrepancy))
        or (self.best_pending_node_info and (smallest_max_discrepancy > self.best_pending_node_info.smallest_max_discrepancy)) then

        new_node_idx = #self.nodes + 1

        local new_node = dqn.Node{s=s, idx=new_node_idx, visit_pct_mom=self.partition_visit_rate_mom}

        self.pending_node_info = {}
        self.pending_node_info.node = new_node
        self.pending_node_info.pending_time = 0
        --self.pending_node_info.rule_out_reason = rule_out_reason
        self.pending_node_info.directions_to_current_state = directions_to_current_state
        self.pending_node_info.self_directions = self_nuvelty
        self.pending_node_info.episode_age = #self.current_episode
        self.pending_node_info.self_error = math.abs(self_novelty)
        self.pending_node_info.smallest_max_discrepancy = smallest_max_discrepancy
        self.pending_node_info.closest_node = self.current_node_num       
        self.pending_node_info.closest_node_visited = 0
        self.pending_node_info.visited = 0
        self.pending_node_info.eval_time = 0
        self.pending_node_info.dir_to = directions_to_current_state
        self.pending_node_info.dir_from = directions_from_current_state

        self.nodes_visited[new_node_idx] = true -- This will get wiped by "addEpisodeToExperienceCache" if a life is soon lost and the node isn't added
    end

    local visits = 0

    if self.current_node_num > 0 then

        self.current_origin_state = self.nodes[self.current_node_num].s:clone()
        visits = self.nodes[self.current_node_num].visits

        if self.show_display then
            self.current_debug_state = self.nodes[self.current_node_num].s:clone()
            self.current_debug_state = image.scale(self.current_debug_state, 160, 210) -- confusing, width and height and back-to-front...
            self.current_debug_state:resize(210, 160)
        end
    end

    self.current_debug_caption = "V: " .. string.format("%.3f", visits)
        .. ", Q: " .. string.format("%.5f", self.bestq)
        .. ", E: " .. string.format("%.3f", self.ep)
        .. ", D: " .. string.format("%.3f", smallest_max_discrepancy)
        .. ", FSR: " .. self.frames_since_reward
end

function nql:get_subgoal_diff(args)

    local current_state = args.s
    
    local candidate_state = args.pending_s

    local direction_difference = {}

    direction_difference[1] = RND_calc_novelty_between_two_states{from=current_state, to=candidate_state}
    
    direction_difference[2] = RND_calc_novelty_between_two_states{from=candidate_state, to=current_state}

    return direction_difference
end

function nql:add_new_node()

    local new_node = self.best_pending_node_info.node

    self.nodes[new_node.idx] = new_node
    
    local avg = 0
    local sd = 0
    
    if self.node_distances[self.best_pending_node_info.closest_node] then
        print("-----------------")
        print("Added node " .. new_node.idx)
        print("Closest node: " .. self.best_pending_node_info.closest_node)
        print("Distance: " .. self.best_pending_node_info.smallest_max_discrepancy)
        
        
        local sum = 0
        local count = 0
        
        for k,v in pairs(self.node_distances[self.best_pending_node_info.closest_node]) do
            sum = sum + v
            count = count + 1
        end

        avg = sum / count
        print("Average distance to closest node: " .. avg)
        
        
        sum = 0
        count = 0
        for k,v in pairs(self.node_distances[self.best_pending_node_info.closest_node]) do
              diff = v - avg
              sum = sum + (diff * diff)
              count = count + 1
        end

        sd = math.sqrt(sum / count)
        print("SD: " .. sd)
        print("Avg + 2*SD: " .. avg + 2*sd)
        
        
        print("Evaluation time: " .. self.best_pending_node_info.eval_time)
        print("Times visited: " .. self.best_pending_node_info.visited)
        print("Times closest node visited: " .. self.best_pending_node_info.closest_node_visited)
        
        print("-----------------")
        
        
    end
    
    if node_distances then
    --Reset stored distances and add new subgoal to table
        for key,value in pairs(node_distances) do
            node_distances[key] = {}
        end        
    end
    
    self.node_distances[new_node.idx] = {}
    
    self:log("\n(+) Creating node " .. new_node.idx)

    self:log("Episode age = " .. self.best_pending_node_info.episode_age)
    self:log("Smallest max. discrepancy = " .. self.best_pending_node_info.smallest_max_discrepancy)
    self:log("-----------------") 
    self:log("Closest node: " .. self.best_pending_node_info.closest_node)
    self:log("Distance: " .. self.best_pending_node_info.smallest_max_discrepancy)
    self:log("Average distance to closest node: " .. avg)
    self:log("SD: " .. sd)
    self:log("Avg + 2*SD: " .. avg + 2*sd)
    self:log("Evaluation time: " .. self.best_pending_node_info.eval_time)
    self:log("Times visited: " .. self.best_pending_node_info.visited)
    self:log("Times closest node visited: " .. self.best_pending_node_info.closest_node_visited)
    self:log("-----------------")

    --[[
    for i = 1, #self.best_pending_node_info.rule_out_reason do
        self:log("Node " .. i .. ":")
        self:log("Ruled out by node " .. self.best_pending_node_info.rule_out_reason[i][1])
        self:log("Direction = " .. self.best_pending_node_info.rule_out_reason[i][2])
        self:log(dump_to_str(self.best_pending_node_info.rule_out_reason[i][3]))
        self:log(dump_to_str(self.best_pending_node_info.rule_out_reason[i][4]))
    end
    --]]

    -- Update directions for all nodes
    --lars RND update dist med novelty (usikker men jeg tror den er der)
    -- Existing nodes to new node
    for i = 1, #self.nodes - 1 do
        self.nodes[i]:add_directions_to_node{to_node=new_node.idx, directions=self.best_pending_node_info.directions_to_current_state[i]}
    end

    new_node:refresh_directions_to_nodes{nodes=self.nodes, pred=self.RND_P_network, targ=self.RND_T_network, gpu=self.gpu}

    new_node:save_img(self.logfilePath)

    self.best_pending_node_info = nil
    self.pending_node_info = nil
end


function nql:handle_pending_node()

    if self.pending_node_info then

        if self.pending_node_info.pending_time > self.episode_end_buffer then

            self.best_pending_node_info = self.pending_node_info
            self.pending_node_info = nil

            print("New max = " .. self.best_pending_node_info.smallest_max_discrepancy)

        else
            self.pending_node_info.pending_time = self.pending_node_info.pending_time + 1
        end
    end

    -- Add a new node if it's time
    self.time_since_node_addition = self.time_since_node_addition + 1

    if self.best_pending_node_info and ((#self.nodes == 0) or (self.time_since_node_addition > self.node_addition_time_gap)) then

        if #self.nodes > 0 then
            self.node_addition_time_gap = self.node_addition_time_gap * self.partition_add_time_mult
        end

        self:add_new_node()
        self.time_since_node_addition = 0
    end
end


function nql:refresh_stored_displacements()

    for i = 1, #self.nodes do
        self.nodes[i]:refresh_directions_to_nodes{nodes=self.nodes, pred=self.RND_P_network, targ=self.RND_T_network, gpu=self.gpu}
    end
end


function nql:update_image_compare_debug_output(args)

    local terminal = args.terminal

    if #self.nodes == 0 and (self.show_display or self.print_action_probs) then

        if not self.current_origin_state
            or terminal
            or self.debug_state_change_steps > self.debug_state_change_rate then

            local s1 = self.transitions:sample_random_valid_frame() or self.current_origin_state

            if s1 then

                self.current_origin_state = s1:clone()

                if self.show_display then
                    self.current_debug_state = s1:clone():resize(84, 84)
                    self.current_debug_state = image.scale(self.current_debug_state, 160, 210) -- confusing, width and height and back-to-front...
                    --self.current_debug_state:resize(210, 160)
                    self.debug_state_change_steps = 0
                end
            end
        end
    end

    if self.print_action_probs then

        self.distance_debug_window:clear()

        local s = (args.s):clone():reshape(1, 84, 84)

        self.image_compare_network:evaluate()

        local stateCopyTensor = torch.FloatTensor(1, 1, 84, 84):fill(0)
        local targetTensor = torch.FloatTensor(1, 1, 84, 84):fill(0)

        stateCopyTensor[1] = self.current_origin_state:clone():resize(1, 1, 84, 84)
        targetTensor[1] = s:clone():resize(1, 1, 84, 84)

        local mergedStates = self:merge_distance_net_states{s1=stateCopyTensor, s2=targetTensor}
        mergedStates:resize(1, 2, 84, 84)

        local net_output = self.image_compare_network:forward(mergedStates):float()

        local errTrain = self:getSumSqErr()

        self.distance_debug_window:add_text("Train err: " .. string.format("%.5f", errTrain))

        self.distance_debug_window:add_text("self.lr_image_compare = " .. self.lr_image_compare)

        self.distance_debug_window:add_text("Half 1:")
        for i, m in ipairs(self.image_compare_network:get(2):get(1).modules) do

            if torch.type(m):find('Convolution') or torch.type(m):find('Linear') then

                local w_mean = m.weight:clone():float():abs():mean()
                local dw_mean = m.gradWeight:clone():float():abs():mean()
                local out_mean = m.output:clone():float():abs():mean()

                self.distance_debug_window:add_text(tostring(i) .. ": w_mean = " .. tostring(w_mean) .. ", dw_mean = " .. tostring(dw_mean) .. ", out_mean = " .. tostring(out_mean))
            end
        end

        self.distance_debug_window:add_text("Half 2:")
        for i, m in ipairs(self.image_compare_network:get(2):get(2).modules) do

            if torch.type(m):find('Convolution') or torch.type(m):find('Linear') then

                local w_mean = m.weight:clone():float():abs():mean()
                local dw_mean = m.gradWeight:clone():float():abs():mean()
                local out_mean = m.output:clone():float():abs():mean()

                self.distance_debug_window:add_text(tostring(i) .. ": w_mean = " .. tostring(w_mean) .. ", dw_mean = " .. tostring(dw_mean) .. ", out_mean = " .. tostring(out_mean))
            end
        end

        self.distance_debug_window:add_text("Remainder:")
        for i, m in ipairs(self.image_compare_network.modules) do

            if torch.type(m):find('Convolution') or torch.type(m):find('Linear') then

                local w_mean = m.weight:clone():float():abs():mean()
                local dw_mean = m.gradWeight:clone():float():abs():mean()
                local out_mean = m.output:clone():float():abs():mean()

                self.distance_debug_window:add_text(tostring(i) .. ": w_mean = " .. tostring(w_mean) .. ", dw_mean = " .. tostring(dw_mean) .. ", out_mean = " .. tostring(out_mean))
            end
        end

        for j = 1, self.n_actions do
            self.distance_debug_window:add_text(self.action_names[j] .. " = " .. string.format("%.5f", net_output[1][j] / self.reward_scale))
        end

        self.image_compare_network:training()

        self.distance_debug_window:repaint()
    end
end


function nql:addEpisodeToExperienceCache(replay_memory, episode)

    local mc_returns = {}
    local reward_idx = 3
    local was_greedy_idx = 7
    local novelty_idx = 8
    local node_visits_idx = 10

    if #self.nodes >= self.q_learn_min_partitions then
        self.episode_count = self.episode_count + 1
    end

    self.dump_counter = self.dump_counter + 1

    local nodes_visited_full_ep = {}
    for i = 1, self.max_nodes do
        nodes_visited_full_ep[i] = false
    end

    -- Update visits and fix visits array within the end buffer
    for frame_num = 1, #episode do

        -- Fix visits array
        if (#episode - frame_num) <= self.episode_end_buffer then
            if frame_num == 1 then
                episode[frame_num][node_visits_idx] = {}
            else
                episode[frame_num][node_visits_idx] = table_deep_copy(episode[frame_num - 1][node_visits_idx])
            end
        end

        local node_visits = episode[frame_num][node_visits_idx]
        for i = 1, self.max_nodes do
            if node_visits[i] then
                nodes_visited_full_ep[i] = true
            end
        end

        local previous_node_visits = {}
        if frame_num > 1 then
            previous_node_visits = episode[frame_num - 1][node_visits_idx]
        end

        --[[
        for i = 1, self.max_nodes do

            if node_visits[i] and not previous_node_visits[i] then

                -- Update node's count
                if self.nodes[i] then
                    self.nodes[i].visits = self.nodes[i].visits + 1
                end
            end
        end
        --]]
    end

    -- Update visits
    for i = 1, self.max_nodes do
        if self.nodes[i] then
            self.nodes[i]:update_visits(nodes_visited_full_ep[i], self.episode_count)
        end
    end


    -- Backpropagate the remainder of the episode's (extrinsic) returns
    mc_returns[#episode] = episode[#episode][reward_idx]

    for frame_num = #episode - 1, 1, -1 do
        mc_returns[frame_num] = episode[frame_num][reward_idx] + self.discount * mc_returns[frame_num + 1]
    end


    --print("-----")
    local q_learning_allowed

    for frame_num = 1, #episode do

        q_learning_allowed = true
        if self.pseudocount_rewards_on
            and ((#self.nodes < self.q_learn_min_partitions) or (frame_num == #episode)) then
            q_learning_allowed = false
        end

        local s, a, r, t, p, _, was_greedy, _, action_probs, node_visits = episode[frame_num][1], episode[frame_num][2], episode[frame_num][3], episode[frame_num][4], episode[frame_num][5], episode[frame_num][6], episode[frame_num][7], episode[frame_num][8], episode[frame_num][9], episode[frame_num][10], episode[frame_num][11]

        -- For debugging
        --[[
        if self.dump_counter == 20 then
            local img = s:clone():resize(84, 84)
            image.save(self.logfilePath .. "img_" .. frame_num .. "_" .. string.format("%.3f", r) .. "_" .. episode[frame_num][node_num_idx] .. ".png", img)
        end
        --]]

        replay_memory:add(s, a, r, t, p, mc_returns[frame_num], frame_num, #episode - frame_num, was_greedy, q_learning_allowed, action_probs, node_visits)
        --print("Frame " .. frame_num .. ", a = " .. a .. ", n_idx = " .. n_idx .. ", r = " .. r .. ", mc r = " .. mc_returns[frame_num])

        --print("frame_num = " .. frame_num .. ", a = " .. a .. ", q_learning_allowed = " .. tostring(q_learning_allowed))
    end

    --print("Ep length = " .. #mc_returns .. ", MC return from ep start = " .. (mc_returns[1] or "N/A"))

    --[[
    --if mc_returns[first_set_frame] < 0 then
        print("Episode dump:")
        for frame_num = 1, #episode do
            print("frame_num = " .. frame_num .. ", node = " .. episode[frame_num][node_num_idx] .. ", r (post clip) = " .. episode[frame_num][reward_idx] .. ", mc re = " .. mc_returns[frame_num])
        end
    --end
    --]]
end


function nql:perceive(reward, rawstate, terminal, testing, testing_ep, first_screen_raw)

    --self:update_distance_stats()

    if not self.first_screen then
        self.first_screen = self:preprocess(first_screen_raw):float()
    end

    self:check_for_settings_files()

    local state = self:preprocess(rawstate):float()

    self.episode_score = self.episode_score + reward

    if reward ~= 0 then
        self.frames_since_reward = 0
    else
        self.frames_since_reward = self.frames_since_reward + 1
    end

    if self.max_reward then
        reward = math.min(reward, self.max_reward)
    end
    if self.min_reward then
        reward = math.max(reward, self.min_reward)
    end
    if self.rescale_r then
        self.r_max = math.max(self.r_max, reward)
    end
    
    --lars: jeg tror ikke det har skal være efter vi har implementeret RND
    --[[]
    if self.numSteps > 0 and self.numSteps % self.stored_displacements_refresh_steps == 0 then
        self:refresh_stored_displacements()
    end
    --]]

    -- Log node visits and thresholds
    if self.numSteps > 0 and self.numSteps % self.logger_steps == 0 and self.nodes then

        local node_visitations_str = ""
        for j = 1, #self.nodes do
            node_visitations_str = node_visitations_str .. j .. ": " .. string.format("%.2f", self.nodes[j].visits) .. ", "
        end

        local average_ep_score = self:getAverageEpisodeScore()
        local errTrain, errTest = self:getSumSqErr()

        --[[
        local distance_scale_str = ""
        if #self.nodes >= 2 then
            distance_scale_str = ", Scale node 1 -> 2 = " .. self.nodes[1].distance_scale[2]
        end

        self:log("Steps: " .. self.numSteps .. ", ep count: " .. self.episode_count .. ", average episode (life) score = " .. average_ep_score .. ", errTrain = " .. errTrain .. ", errTest = " .. errTest .. distance_scale_str)
        --]]

        ---[[
        self:log("Node visits: " .. node_visitations_str)
        --]]

        if self.dump_convolutional_filters then
            local filter_weights = self.image_compare_network:get(2):get(1):get(2).weight
            for i = 1, filter_weights:size(1) do
                local img = image.scale(filter_weights[i][1]:clone():float(), 240, 240, 'simple')
                image.save(self.logfilePath .. "filter" .. i .. ".png", img)
            end
        end
    end

    -- Update current node, create new nodes, handle visits, etc
    self.transitions:add_recent_state_ic(state, terminal)

    self.node_change_counter = self.node_change_counter + 1

    if terminal then

        if self.current_debug_state then
            self.current_debug_state:zero()
        end

        -- This is just to ensure that we don't start the node creation process in the middle of an episode
        if self.numSteps > self.partition_creation_start then
            self.image_compare_node_creation_begun = true
        end

        self.episode_scores[self.episode_scores_idx] = self.episode_score
        self.episode_scores_idx = self.episode_scores_idx % self.episode_scores_averaging_n + 1
        self.episode_score = 0
    end

    -- Reset at the start of each new episode
    if terminal or self.lastTerminal then
        self.current_node_num = -2
        self.pending_node_info = nil
    end

    --Store transition s, a, r, s'
    if self.lastState and not testing then

        -- ##### TO DO: Tidy this up. At least get rid of unused 'nil' values #####
        self.current_episode[#self.current_episode + 1] = {self.lastState:clone(), self.lastAction, reward, self.lastTerminal, priority, self.last_node_num, self.lastActionGreedy, nil, self.lastActionProbs, self.last_nodes_visited}
    end

    -- At the end of each episode, add it to the cache
    if self.lastTerminal then

        self:addEpisodeToExperienceCache(self.transitions, self.current_episode)

        self.current_episode                = {}
        self.nodes_visited                  = {}
        self.last_nodes_visited             = {}
        self.frames_since_reward            = 0
        self.frames_since_reward_exceeded   = false
        self.bg_visited                     = false
        self.first_pass                     = true
    end

    -- For debugging episode start buffer
    --[[
    local dump_frame_num = 0
    if #self.current_episode == dump_frame_num then
        image.save(self.logfilePath .. "frame_" .. dump_frame_num .. ".png", state:clone():resize(84,84))
    end
    --]]

    if self.pseudocount_rewards_on
        and self.image_compare_node_creation_begun
        and not terminal
        and ((self.node_change_counter >= self.partition_update_freq) or (self.current_node_num == -2)) then

        self:update_nodes{s=state}
        self.node_change_counter = 0 -- ##### Shouldn't do if just returned! #####
    end

    if self.pseudocount_rewards_on then
        self:handle_pending_node()
    end

    self.transitions:add_recent_state(state, terminal)

    -- currentFullState includes additional recent frames so that the agent can judge velocities, etc
    local currentFullState = self.transitions:get_recent()

    if terminal or self.numSteps % self.partition_update_freq == 0 or not self.current_debug_state then
        self:update_image_compare_debug_output{s=state, terminal=terminal}
    end

    if self.numSteps == (self.learn_start + 1) and not testing then
        self:sample_validation_data()
    end

    local curState = self.transitions:get_recent():reshape(1, 4 * 84 * 84)

-- zzzzz
    local node_visit_info = torch.FloatTensor(1, self.max_nodes):fill(0)

    for i = 1, self.max_nodes do
        if self.nodes_visited[i] then
            if i > #self.nodes then node_visit_info[1][i] = self.ee_beta
            else node_visit_info[1][i] = self.ee_beta / math.sqrt(math.max(1, self.nodes[i].visits)) end
        end
    end

    curState = torch.cat(curState, node_visit_info, 2)

    -- Select action
    local actionIndex = 1
    local was_greedy = true

    local action_probs = {}
    action_probs[1] = 1
    for i = 2, self.n_actions do
        action_probs[i] = 0
    end

    if not terminal then
        actionIndex, was_greedy, action_probs = self:get_action(curState, testing_ep)
    end

    --print("action_probs[actionIndex] = " .. action_probs[actionIndex])

    -- For testing skull dependence
    --[[
    if self.numSteps > 1000 then
        actionIndex = 1
    end
    --]]

    self.transitions:add_recent_action(actionIndex)
    self.transitions:add_recent_action_ic(actionIndex)

    --Do some Q-learning updates
    if self.numSteps > self.learn_start
        and not testing
        and self.numSteps % self.update_freq == 0 then

        for i = 1, self.n_replay do
            self:qLearnMinibatch()
        end
    end

    --Do some RND updates
    if self.transitions:size() >= self.RND_learn_start
        and self.numSteps <= self.RND_learn_end
        and not testing
        and self.numSteps % self.image_compare_update_freq == 0 then

        if self.step_count_when_learning_began == -1 then
            self.step_count_when_learning_began = self.numSteps
        end

        for i = 1, self.RND_n_replay do
            self:RND_update()
        end
    end

    if not testing then
        self.numSteps = self.numSteps + 1
    end

    self.debug_state_change_steps = self.debug_state_change_steps + 1

    self.lastState = state:clone()
    self.lastAction = actionIndex
    self.lastActionGreedy = was_greedy
    self.lastActionProbs = action_probs
    self.lastTerminal = terminal
    self.last_node_num = self.current_node_num
    self.last_nodes_visited = table_deep_copy(self.nodes_visited)

    if self.target_q and self.numSteps % self.target_q == 1 then
        self.target_network = self.network:clone()
    end

    -- Display screen, along with some debugging info
    if self.show_display then

        if self.useRGB then
            self.display_image = rawstate:clone():float()
        else
            if not self.current_debug_state then
                self.current_debug_state = rawstate:clone():float():zero()
            end

            self.display_image = torch.cat(rawstate:clone():float():mul(255), self.current_debug_state:clone():mul(255), 2)
        end

        self.win_main = image.display({image=self.display_image, win=self.win_main, legend=self.current_debug_caption})
    end

    if not terminal then
        return actionIndex
    else
        return 0
    end
end


function nql:set_epsilon(state, testing_ep)

    if testing_ep then
        self.ep = testing_ep
    elseif self.frames_since_reward > self.non_reward_frames_before_full_eps then
        self.ep = self.ep_start
    elseif self.numSteps <= self.learn_start then
        self.ep = self.ep_start
    else
        self.ep = self.ep_end + math.max(0, (self.ep_start - self.ep_end) * (self.ep_endt - math.max(0, self.numSteps - self.learn_start)) / self.ep_endt)
    end
end


function nql:greedy(q)

    local maxq = q[1]
    local besta = {1}

    -- Evaluate all other actions (with random tie-breaking)
    for a = 2, self.n_actions do
        if q[a] > maxq then
            besta = { a }
            maxq = q[a]
        elseif q[a] == maxq then
            besta[#besta + 1] = a
        end
    end
    self.bestq = maxq

    local r = 1
    if #besta > 1 then
        r = torch.random(1, #besta)
    end

    local action_probs = {}
    for i = 1, self.n_actions do

        action_probs[i] = 0

        for j = 1, #besta do
            if besta[j] == i then
                action_probs[i] = 1 / #besta
            end
        end
    end

    return besta[r], true, action_probs
end


function nql:eGreedy(q)

    local actionIndex, was_greedy, action_probs = self:greedy(q)
    for i = 1, self.n_actions do
        action_probs[i] = action_probs[i] * (1 - self.ep) + self.ep / self.n_actions
    end

    if torch.uniform() < self.ep then
        actionIndex = torch.random(1, self.n_actions)
        was_greedy = false
    end

    return actionIndex, was_greedy, action_probs
end


function nql:softmax(q)

    local action_probs = {}
    local minq = math.huge
    local maxq = -math.huge

    for i = 1, self.n_actions do

        if q[i] < minq then
            minq = q[i]
        end

        if q[i] > maxq then
            maxq = q[i]
        end
    end
    self.bestq = maxq

    local rangeq = maxq - minq
    local maxToMinPickRatio = ((1 - self.ep) + self.ep / self.n_actions) / (self.ep / self.n_actions)
    local softmaxT = math.log(maxToMinPickRatio) / (rangeq + 0.00001) -- Extra constant added for stability
    local maxExponent = softmaxT * maxq
    local weights = {}
    local cumuWeights = {}
    local sumWeights = 0

    for i = 1, self.n_actions do
        weights[i] = math.exp(softmaxT * q[i] - maxExponent)
        cumuWeights[i] = (cumuWeights[i - 1] or 0) + weights[i]
        sumWeights = sumWeights + weights[i]
    end

    for i = 1, self.n_actions do
        action_probs[i] = weights[i] / sumWeights
    end

    local cutoff = torch.uniform() * sumWeights
    local actionSelected = -1

    for i = 1, self.n_actions do
        if cumuWeights[i] > cutoff then
            actionSelected = i
            --print("Selected action " .. i .. ", prob = " .. weights[i] / sumWeights)
            break
        end
    end

    return actionSelected, false, action_probs
end


function nql:get_action(state, testing_ep)

    -- For debugging
    --[[
    if self.numSteps % 100 == 0 then
        for i = 1, state:size(2) do
            print(i .. ": " .. state[1][i])
        end
    end
    --]]

    self:set_epsilon(state, testing_ep)

    -- Turn single state into minibatch.  Needed for convolutional nets.
    --[[
    if state:dim() == 2 then
        assert(false, 'Input must be at least 3D')
        state = state:resize(1, state:size(1), state:size(2))
    end
    --]]

    if self.gpu >= 0 then
        state = state:cuda()
    end

    local q = self.network:forward(state):float():squeeze()

    if self.exploration_style == "eGreedy" then
        return self:eGreedy(q)

    elseif self.exploration_style == "softmax" then
        return self:softmax(q)

    else
        error("Unknown training method!")

    end
end


function nql:report()
    print(get_weight_norms(self.network))
    print(get_grad_norms(self.network))
end

function nql:RND_calc_novelty(next_state)
    -- this function calculates the novelty of a state
    local criterion = nn.MSECriterion()
    local prediction = self.RND_P_network:forward(next_state)
    local target = self.RND_T_network:forward(next_state)
    return criterion:forward(prediction, target)
end

function nql:RND_update()
    assert(self.transitions:size() > self.minibatch_size)

    local s,s2 = self.transitions:sample_RND(self.minibatch_size)

    net = self.RND_P_network
    s:reshape(self.minibatch_size, 1, 84,  84)
    s2:reshape(self.minibatch_size, 1, 84,  84)
    local net_input = self:merge_distance_net_states{s1=s, s2=s2}
    
    targ = self.RND_T_network:forward(net_input):clone():float()
    pred = self.RND_P_network:forward(net_input):clone():float()
    -- compute linearly annealed learning rate
    local t = math.max(0, self.numSteps - self.step_count_when_learning_began)
    self.lr_image_compare = (self.lr_start_image_compare - self.lr_end_image_compare) * (self.lr_endt_image_compare - t) / self.lr_endt_image_compare + self.lr_end_image_compare
    self.lr_image_compare = math.max(self.lr_image_compare, self.lr_end_image_compare)

    self.config_adam.learningRate = self.lr_image_compare
    self.config_adam.momentum = self.mom_image_compare
    
    local w = self.RND_w
    local dw = self.RND_dw

    -- add weight cost to gradient
    function feval(params)
        
        -- zero gradients of parameters
        dw:zero()

        local loss = nn.MSECriterion():forward(pred, targ)

        local dloss_doutputs = nn.MSECriterion():backward(pred, targ)
        if self.gpu >= 0 then
            dloss_doutputs = dloss_doutputs:cuda()
        end

        -- get new gradient
        net:backward(net_input, dloss_doutputs)

        return loss, dw
    end

    optim.adam(feval, w, self.config_adam)
end

function nql:RND_calc_novelty_between_two_states(args)
    local net_input = torch.FloatTensor(1, 2, 84, 84):fill(0)
    local from_state = args.from
    local to_state = args.to
    net_input[1][1]:copy(from_state)
    net_input[1][2]:copy(to_state)

    if self.gpu and self.gpu >= 0 then
        net_input = net_input:cuda()
    else
        net_input = net_input:float()
    end

    return self:RND_calc_novelty(net_input)
end