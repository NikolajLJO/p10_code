--[[
Copyright (c) 2014 Google Inc.

See LICENSE file for full terms of limited license.
]]

require 'image'

local trans = torch.class('dqn.TransitionTable')


function trans:__init(args)

    self.agent                         = args.agent
    self.stateDim                      = args.stateDim
    self.numActions                    = args.numActions
    self.histLen                       = args.histLen
    self.maxSize                       = args.maxSize or 1024^2
    self.bufferSize                    = args.bufferSize or 1024
    self.histType                      = args.histType or "linear"
    self.histSpacing                   = args.histSpacing or 1
    self.zeroFrames                    = args.zeroFrames or 1
    self.nonTermProb                   = args.nonTermProb or 1
    self.nonEventProb                  = args.nonEventProb or 1
    self.gpu                           = args.gpu
    self.numEntries                    = 0
    self.insertIndex                   = 0
    self.episode_end_buffer            = args.episode_end_buffer
    self.discount                      = args.discount
    self.ee_discount                   = args.ee_discount

    self.histIndices            = {}
    self.ee_histIndices         = {}

    local histLen               = self.histLen

    if self.histType == "linear" then
        -- History is the last histLen frames.
        self.recentMemSize = self.histSpacing*histLen
        for i=1,histLen do
            self.histIndices[i] = i*self.histSpacing
        end
    elseif self.histType == "exp2" then
        -- The ith history frame is from 2^(i-1) frames ago.
        self.recentMemSize = 2^(histLen-1)
        self.histIndices[1] = 1
        for i=1,histLen-1 do
            self.histIndices[i+1] = self.histIndices[i] + 2^(7-i)
        end
    elseif self.histType == "exp1.25" then
        -- The ith history frame is from 1.25^(i-1) frames ago.
        self.histIndices[histLen] = 1
        for i=histLen-1,1,-1 do
            self.histIndices[i] = math.ceil(1.25*self.histIndices[i+1])+1
        end
        self.recentMemSize = self.histIndices[1]
        for i=1,histLen do
            self.histIndices[i] = self.recentMemSize - self.histIndices[i] + 1
        end
    end

    -- History is the last histLen frames.
    self.recentMemSize_ee = self.agent.ee_histSpacing * self.agent.ee_histLen
    for i = 1, self.agent.ee_histLen do
        self.ee_histIndices[i] = i * self.agent.ee_histSpacing
    end

    -- For debugging
    self.image_dump_counter = 1

    self.s                      = torch.ByteTensor(self.maxSize, self.stateDim):fill(0)
    self.a                      = {}
    self.r                      = {}
    self.ret_mc                 = {}
    self.t                      = {}
    self.frames_before_term_adj = {}
    self.action_encodings       = torch.eye(self.numActions)
    self.was_greedy             = {}
    self.q_learning_allowed     = {}
    self.action_probs           = {}
    self.node_visits            = {}


    -- Tables for storing the last histLen states.  They are used for
    -- constructing the most recent agent state more easily.
    self.recent_s = {}
    self.recent_a = {}
    self.recent_t = {}

    self.recent_s_ic = {}
    self.recent_a_ic = {}
    self.recent_t_ic = {}

    self.valid_indices = {}

    local s_size = self.stateDim * histLen

    self.buf_a               = torch.LongTensor(self.bufferSize):fill(0)
    self.buf_r               = torch.zeros(self.bufferSize)
    self.buf_ret_mc          = torch.zeros(self.bufferSize)
    self.buf_term            = torch.ByteTensor(self.bufferSize):fill(0)
    self.buf_s               = torch.ByteTensor(self.bufferSize, s_size):fill(0)
    self.buf_s2              = torch.ByteTensor(self.bufferSize, s_size):fill(0)
    self.buf_node_visits_s   = torch.zeros(self.bufferSize, self.agent.max_nodes)
    self.buf_node_visits_s2  = torch.zeros(self.bufferSize, self.agent.max_nodes)

    if self.gpu and self.gpu >= 0 then
        self.gpu_s  = self.buf_s:float():cuda()
        self.gpu_s2 = self.buf_s2:float():cuda()
    end

    local s_size_ic              = self.stateDim * self.agent.ee_histLen
    self.buf_ic_s                = torch.ByteTensor(self.bufferSize, s_size_ic):fill(0)
    self.buf_ic_s2               = torch.ByteTensor(self.bufferSize, s_size_ic):fill(0)
    self.buf_ic_smid             = torch.ByteTensor(self.bufferSize, s_size_ic):fill(0)
    self.buf_ic_a                = {}
    self.buf_ic_a_count          = {}
    self.buf_ic_action_probs     = {}

    for i = 1, self.bufferSize do
        self.buf_ic_a_count[i] = {}
        self.buf_ic_action_probs[i] = {}
    end

    if self.gpu and self.gpu >= 0 then
        self.gpu_ic_s       = self.buf_ic_s:float():cuda()
        self.gpu_ic_s2      = self.buf_ic_s2:float():cuda()
        self.gpu_ic_smid    = self.buf_ic_smid:float():cuda()
    end
end


function trans:reset()
    self.numEntries = 0
    self.insertIndex = 0
end


function trans:size()
    return self.numEntries
end


function trans:empty()
    return self.numEntries == 0
end


function trans:fill_buffer()
    assert(self.numEntries >= self.bufferSize)
    -- clear CPU buffers
    self.buf_ind = 1
    local ind
    for buf_ind=1,self.bufferSize do
        local s, a, r, s2, term, ret_mc, node_visits_s, node_visits_s2 = self:sample_one(1)
        self.buf_s[buf_ind]:copy(s)
        self.buf_a[buf_ind] = a
        self.buf_r[buf_ind] = r
        self.buf_ret_mc[buf_ind] = ret_mc
        self.buf_s2[buf_ind]:copy(s2)
        self.buf_term[buf_ind] = term
        self.buf_node_visits_s[buf_ind] = node_visits_s
        self.buf_node_visits_s2[buf_ind] = node_visits_s2
    end
    self.buf_s  = self.buf_s:float():div(255)
    self.buf_s2 = self.buf_s2:float():div(255)
    if self.gpu and self.gpu >= 0 then
        self.gpu_s:copy(self.buf_s)
        self.gpu_s2:copy(self.buf_s2)
    end
end


function trans:fill_buffer_ic(args)

    assert(self.numEntries >= self.bufferSize)

    -- clear CPU buffers
    self.buf_ind_ic = 1

    for i = 1, self.bufferSize do
        local s, s2, smid, a, a_count, action_probs = self:sample_one_ee()
        self.buf_ic_s[i]:copy(s)
        self.buf_ic_s2[i]:copy(s2)
        self.buf_ic_smid[i]:copy(smid)
        self.buf_ic_a[i] = a
        self.buf_ic_a_count[i] = a_count
        self.buf_ic_action_probs[i] = action_probs
    end

    self.buf_ic_s = self.buf_ic_s:float():div(255)
    self.buf_ic_s2 = self.buf_ic_s2:float():div(255)
    self.buf_ic_smid = self.buf_ic_smid:float():div(255)

    if self.gpu and self.gpu >= 0 then
        self.gpu_ic_s:copy(self.buf_ic_s)
        self.gpu_ic_s2:copy(self.buf_ic_s2)
        self.gpu_ic_smid:copy(self.buf_ic_smid)
    end
end


function trans:wrap_idx(idx)
    local result = idx % self.maxSize
    if result == 0 then
        result = self.maxSize
    end
    return result
end


function trans:sample_one()

    assert(self.numEntries > 1)
    local index
    local ar_index
    local valid = false
    while not valid do

        -- start at 2 because of previous action
        index = torch.random(2, self.numEntries - self.recentMemSize)
        ar_index = self:wrap_idx(index + self.recentMemSize - 1)

        if (self.t[ar_index] == 0) and self.q_learning_allowed[ar_index] then
            valid = true
        end
    end

    return self:get(index)
end


function trans:sample_one_ee()

    local index = torch.random(1, self.numEntries)
    local ar_index = self:wrap_idx(index + self.recentMemSize_ee - 1)

    while not self.valid_indices[ar_index] do
        index = torch.random(1, self.numEntries)
        ar_index = self:wrap_idx(index + self.recentMemSize_ee - 1)
    end

    local frames_before_term = self.frames_before_term_adj[ar_index]

    local offset = torch.random(1, frames_before_term)
    offset = math.min(offset, self.agent.ee_time_sep_constant_m)

    local secondIndex = self:wrap_idx(index + offset)
    local midIndex = self:wrap_idx(index + 1)

    local s1, s2, smid = self:concatFrames_ic(index), self:concatFrames_ic(secondIndex), self:concatFrames_ic(midIndex)

    local action_rewards_mc = {}

    for i = 1, offset do

        local action_idx = self:wrap_idx(index + self.recentMemSize_ee - 1 + (i - 1))
        local action_taken = self.a[action_idx]

        for j = 1, self.numActions do

            local reward

            if self.agent.ee_comparison_policy == "uniform" then

                if j == action_taken then
                    reward = 1 - (1 / self.numActions)
                else
                    reward = -1 / self.numActions
                end

            elseif self.agent.ee_comparison_policy == "current" then

                if j == action_taken then
                    reward = 1 - self.action_probs[action_idx][j]
                else
                    reward = -self.action_probs[action_idx][j]
                end

            else
                error("Unknown comparison policy!")

            end

            reward = reward * self.agent.reward_scale

            action_rewards_mc[j] = (action_rewards_mc[j] or 0) + (self.ee_discount ^ (i - 1)) * reward
        end
    end

    local action_probs = {}
    for i = 1, self.numActions do
        action_probs[i] = self.action_probs[ar_index][i]
    end

    return s1, s2, smid, self.a[ar_index], action_rewards_mc, action_probs
end


function trans:get_table_range(tbl, range)

    local result = {}
    for i = range[1][1], range[1][2] do
        result[#result + 1] = tbl[i]
    end

    return result
end


function trans:sample(batch_size)
    local batch_size = batch_size or 1
    assert(batch_size < self.bufferSize)

    if not self.buf_ind or self.buf_ind + batch_size - 1 > self.bufferSize then
        self:fill_buffer()
    end

    local index = self.buf_ind

    self.buf_ind = self.buf_ind+batch_size
    local range = {{index, index+batch_size-1}}

    local buf_s, buf_s2, buf_a, buf_r, buf_term, buf_ret_mc, buf_node_visits_s, buf_node_visits_s2 = self.buf_s, self.buf_s2,
        self.buf_a, self.buf_r, self.buf_term, self.buf_ret_mc, self.buf_node_visits_s, self.buf_node_visits_s2
    if self.gpu and self.gpu >=0  then
        buf_s = self.gpu_s
        buf_s2 = self.gpu_s2
    end

    return buf_s[range], buf_a[range], buf_r[range], buf_s2[range], buf_term[range], buf_ret_mc[range], buf_node_visits_s[range], buf_node_visits_s2[range]
end


function trans:sample_ic(args)

    local batch_size = args.batch_size or 1

    assert(batch_size < self.bufferSize)

    if not self.buf_ind_ic or self.buf_ind_ic + batch_size - 1 > self.bufferSize then
        self:fill_buffer_ic()
    end

    local index = self.buf_ind_ic

    self.buf_ind_ic = self.buf_ind_ic + batch_size
    local range = {{index, index + batch_size - 1}}

    local buf_ic_s, buf_ic_s2, buf_ic_smid, buf_ic_a, buf_ic_a_count, buf_ic_action_probs = self.buf_ic_s, self.buf_ic_s2, self.buf_ic_smid, self.buf_ic_a, self.buf_ic_a_count, self.buf_ic_action_probs
    if self.gpu and self.gpu >=0  then
        buf_ic_s = self.gpu_ic_s
        buf_ic_s2 = self.gpu_ic_s2
        buf_ic_smid = self.gpu_ic_smid
    end

    return buf_ic_s[range], buf_ic_s2[range], buf_ic_smid[range], self:get_table_range(buf_ic_a, range), self:get_table_range(buf_ic_a_count, range), self:get_table_range(buf_ic_action_probs, range)
end


function trans:sample_random_valid_frame()

    if self.numEntries == 0 then
        return nil
    end

    local iter = 0
    local max_iter = 100

    local index = torch.random(1, self.numEntries)

    while not self.valid_indices[index] do

        index = torch.random(1, self.numEntries)

        iter = iter + 1

        if iter > max_iter then
            return nil
        end
    end

    local s = self:concatFrames_ic(index):float():div(255)
    return s
end


function trans:sample_valid_n_step_pair(n)

    local index = torch.random(1, self.numEntries)

    while (not self.valid_indices[index]) or ((self.frames_before_term_adj[index] - n) <= 0) do
        index = torch.random(1, self.numEntries)
    end

    local index2 = (index + n - 1) % self.numEntries + 1

    local s = self:concatFrames_ic(index):float():div(255)
    local s2 = self:concatFrames_ic(index2):float():div(255)

    return s, s2
end


function trans:concatFrames(index, use_recent)
    if use_recent then
        s, t = self.recent_s, self.recent_t
    else
        s, t = self.s, self.t
    end

    local fullstate = s[1].new()
    fullstate:resize(self.histLen, unpack(s[1]:size():totable()))

    -- Zero out frames from all but the most recent episode.
    local zero_out = false
    local episode_start = self.histLen

    for i=self.histLen-1,1,-1 do
        if not zero_out then
            for j=index+self.histIndices[i]-1,index+self.histIndices[i+1]-2 do
                if t[j] == 1 then
                    zero_out = true
                    break
                end
            end
        end

        if zero_out then
            fullstate[i]:zero()
        else
            episode_start = i
        end
    end

    if self.zeroFrames == 0 then
        episode_start = 1
    end

    -- Copy frames from the current episode.
    for i=episode_start,self.histLen do
        fullstate[i]:copy(s[index+self.histIndices[i]-1])
    end

    return fullstate
end


function trans:concatFrames_ic(index, use_recent)
    if use_recent then
        s, t = self.recent_s_ic, self.recent_t_ic
    else
        s, t = self.s, self.t
    end

    local fullstate = s[1].new()
    fullstate:resize(self.agent.ee_histLen, unpack(s[1]:size():totable()))

    -- Zero out frames from all but the most recent episode.
    local zero_out = false
    local episode_start = self.agent.ee_histLen

    for i = self.agent.ee_histLen - 1, 1, -1 do
        if not zero_out then
            for j=index+self.ee_histIndices[i]-1,index+self.ee_histIndices[i+1]-2 do
                local adjusted_j = ((j - 1) % self.numEntries) + 1
                if t[adjusted_j] == 1 then
                    zero_out = true
                    break
                end
            end
        end

        if zero_out then
            fullstate[i]:zero()
        else
            episode_start = i
        end
    end

    if self.zeroFrames == 0 then
        episode_start = 1
    end

    -- Copy frames from the current episode.
    for i = episode_start, self.agent.ee_histLen do
        local adjusted_s_idx = (((index + self.ee_histIndices[i] - 1) - 1) % self.numEntries) + 1
        fullstate[i]:copy(s[adjusted_s_idx])
    end

    return fullstate
end


function trans:concatActions(index, use_recent)
    local act_hist = torch.FloatTensor(self.histLen, self.numActions)
    if use_recent then
        a, t = self.recent_a, self.recent_t
    else
        a, t = self.a, self.t
    end

    -- Zero out frames from all but the most recent episode.
    local zero_out = false
    local episode_start = self.histLen

    for i=self.histLen-1,1,-1 do
        if not zero_out then
            for j=index+self.histIndices[i]-1,index+self.histIndices[i+1]-2 do
                if t[j] == 1 then
                    zero_out = true
                    break
                end
            end
        end

        if zero_out then
            act_hist[i]:zero()
        else
            episode_start = i
        end
    end

    if self.zeroFrames == 0 then
        episode_start = 1
    end

    -- Copy frames from the current episode.
    for i=episode_start,self.histLen do
        act_hist[i]:copy(self.action_encodings[a[index+self.histIndices[i]-1]])
    end

    return act_hist
end


function trans:get_recent()
    -- Assumes that the most recent state has been added, but the action has not
    return self:concatFrames(1, true):float():div(255)
end


function trans:get_recent_ic()
    -- Assumes that the most recent state has been added, but the action has not
    return self:concatFrames_ic(1, true):float():div(255)
end


function trans:get(index)

    local s = self:concatFrames(index)
    local s2 = self:concatFrames(index + 1)
    local ar_index = self:wrap_idx(index + self.recentMemSize - 1)
    local next_ar_index = self:wrap_idx(ar_index + 1)

    local node_visits_s = torch.zeros(self.agent.max_nodes)
    local node_visits_s2 = torch.zeros(self.agent.max_nodes)
    
    -- Calculate pseudo rewards at sample time
    local nodes = self.agent.nodes

-- zzzzz
    for i = 1, self.agent.max_nodes do
        
        if self.node_visits[ar_index][i] then
            if i > #nodes then node_visits_s[i] = self:calculate_novelty(0)
            else
                node_visits_s[i] = self:calculate_novelty(nodes[i].visits)
            end
        end

        if self.node_visits[next_ar_index][i] then
            if i > #nodes then node_visits_s2[i] = self:calculate_novelty(0)
            else
                node_visits_s2[i] = self:calculate_novelty(nodes[i].visits)
            end           
        end
    end

    

    local full_reward = self.r[ar_index]
    local full_mc_return = self.ret_mc[ar_index]

    local s1_idx = ar_index
    local s2_idx = next_ar_index

    if self.agent.pseudocount_rewards_on then

        local pellet_reward

        for k, v in pairs(self.node_visits[s2_idx]) do
            if not self.node_visits[s1_idx][k] and nodes[k] then

                pellet_reward = self:calculate_novelty(nodes[k].visits)

                -- Clip
                pellet_reward = math.min(self.agent.max_pellet_reward, pellet_reward)

                full_reward = full_reward + pellet_reward
            end
        end

        local n_steps = 0
        while self.t[s2_idx] ~= 1 do

            for k, v in pairs(self.node_visits[s2_idx]) do
                if not self.node_visits[s1_idx][k] and nodes[k] then

                    pellet_reward = self:calculate_novelty(nodes[k].visits)
					
					-- Clip (important to do this before discounting!)
					pellet_reward = math.min(self.agent.max_pellet_reward, pellet_reward)
					
					-- Discount
					pellet_reward = pellet_reward * (self.discount ^ n_steps)

                    full_mc_return = full_mc_return + pellet_reward
                end
            end

            n_steps = n_steps + 1
            s1_idx = s1_idx % self.maxSize + 1
            s2_idx = s2_idx % self.maxSize + 1
        end
    end

    return s, self.a[ar_index], full_reward, s2, self.t[next_ar_index], full_mc_return, node_visits_s, node_visits_s2
end


function trans:calculate_novelty(visits)

    local pseudo_reward = self.agent.ee_beta / math.sqrt(math.max(1, visits))
    return pseudo_reward
end


-- p (the priority) is not actually used
function trans:add(s, a, r, term, p, ret_mc, episode_age, frames_before_term, was_greedy, q_learning_allowed, action_probs, node_visits)

    assert(s, 'State cannot be nil')
    assert(a, 'Action cannot be nil')
    assert(r, 'Reward cannot be nil')

    -- Incremenet until at full capacity
    if self.numEntries < self.maxSize then
        self.numEntries = self.numEntries + 1
    end

    -- Always insert at next index, then wrap around
    self.insertIndex = self.insertIndex + 1
    -- Overwrite oldest experience once at capacity
    if self.insertIndex > self.maxSize then
        self.insertIndex = 1
    end

    -- Overwrite (s,a,r,t) at insertIndex
    self.s[self.insertIndex] = s:clone():float():mul(255)
    self.a[self.insertIndex] = a
    self.r[self.insertIndex] = r
    if term then
        self.t[self.insertIndex] = 1
    else
        self.t[self.insertIndex] = 0
    end

    --local frames_before_term_adjusted = frames_before_term - self.episode_end_buffer - ((self.histLen - 1) * self.recentMemSize)
    local frames_before_term_adjusted = frames_before_term - self.episode_end_buffer

    self.frames_before_term_adj[self.insertIndex] = frames_before_term_adjusted
    self.q_learning_allowed[self.insertIndex] = q_learning_allowed
    self.action_probs[self.insertIndex] = action_probs
    self.node_visits[self.insertIndex] = node_visits

    if was_greedy then
        self.was_greedy[self.insertIndex] = 1
    else
        self.was_greedy[self.insertIndex] = 0
    end

    if frames_before_term_adjusted > 0 then
        self.valid_indices[self.insertIndex] = self.insertIndex
    else
        self.valid_indices[self.insertIndex] = nil
    end

    self.ret_mc[self.insertIndex] = ret_mc
end


function trans:add_recent_state(s, term)

    local s = s:clone():float():mul(255):byte()
    if #self.recent_s == 0 then
        for i = 1, self.recentMemSize do
            table.insert(self.recent_s, s:clone():zero())
            table.insert(self.recent_t, 1)
        end
    end

    table.insert(self.recent_s, s)
    if term then
        table.insert(self.recent_t, 1)
    else
        table.insert(self.recent_t, 0)
    end

    -- Keep recentMemSize states.
    if #self.recent_s > self.recentMemSize then
        table.remove(self.recent_s, 1)
        table.remove(self.recent_t, 1)
    end
end


function trans:add_recent_state_ic(s, term)
    local s = s:clone():float():mul(255):byte()
    if #self.recent_s_ic == 0 then
        for i = 1, self.recentMemSize_ee do
            table.insert(self.recent_s_ic, s:clone():zero())
            table.insert(self.recent_t_ic, 1)
        end
    end

    table.insert(self.recent_s_ic, s)
    if term then
        table.insert(self.recent_t_ic, 1)
    else
        table.insert(self.recent_t_ic, 0)
    end

    -- Keep recentMemSize_ee states.
    if #self.recent_s_ic > self.recentMemSize_ee then
        table.remove(self.recent_s_ic, 1)
        table.remove(self.recent_t_ic, 1)
    end
end


function trans:add_recent_action(a)
    if #self.recent_a == 0 then
        for i=1,self.recentMemSize do
            table.insert(self.recent_a, 1)
        end
    end

    table.insert(self.recent_a, a)

    -- Keep recentMemSize steps.
    if #self.recent_a > self.recentMemSize then
        table.remove(self.recent_a, 1)
    end
end


function trans:add_recent_action_ic(a)
    if #self.recent_a_ic == 0 then
        for i=1,self.recentMemSize_ee do
            table.insert(self.recent_a_ic, 1)
        end
    end

    table.insert(self.recent_a_ic, a)

    -- Keep recentMemSize_ee steps.
    if #self.recent_a_ic > self.recentMemSize_ee then
        table.remove(self.recent_a_ic, 1)
    end
end


--[[
Override the write function to serialize this class into a file.
We do not want to store anything into the file, just the necessary info
to create an empty transition table.

@param file (FILE object ) @see torch.DiskFile
--]]
function trans:write(file)
    file:writeObject({self.stateDim,
                      self.numActions,
                      self.histLen,
                      self.maxSize,
                      self.bufferSize,
                      self.numEntries,
                      self.insertIndex,
                      self.recentMemSize,
                      self.histIndices})
end


--[[
Override the read function to desearialize this class from file.
Recreates an empty table.

@param file (FILE object ) @see torch.DiskFile
--]]
function trans:read(file)
    local stateDim, numActions, histLen, maxSize, bufferSize, numEntries, insertIndex, recentMemSize, histIndices = unpack(file:readObject())
    self.stateDim = stateDim
    self.numActions = numActions
    self.histLen = histLen
    self.maxSize = maxSize
    self.bufferSize = bufferSize
    self.recentMemSize = recentMemSize
    self.histIndices = histIndices
    self.numEntries = 0
    self.insertIndex = 0

    self.s                      = torch.ByteTensor(self.maxSize, self.stateDim):fill(0)
    self.a                      = {}
    self.r                      = {}
    self.ret_mc                 = {}
    self.t                      = {}
    self.frames_before_term_adj = {}
    self.action_encodings       = torch.eye(self.numActions)

    -- Tables for storing the last histLen states.  They are used for
    -- constructing the most recent agent state more easily.
    self.recent_s = {}
    self.recent_a = {}
    self.recent_t = {}

    self.buf_a        = torch.LongTensor(self.bufferSize):fill(0)
    self.buf_r        = torch.zeros(self.bufferSize)
    self.buf_ret_mc   = torch.zeros(self.bufferSize)
    self.buf_term     = torch.ByteTensor(self.bufferSize):fill(0)
    self.buf_s        = torch.ByteTensor(self.bufferSize, self.stateDim * self.histLen):fill(0)
    self.buf_s2       = torch.ByteTensor(self.bufferSize, self.stateDim * self.histLen):fill(0)

    if self.gpu and self.gpu >= 0 then
        self.gpu_s  = self.buf_s:float():cuda()
        self.gpu_s2 = self.buf_s2:float():cuda()
    end
end
