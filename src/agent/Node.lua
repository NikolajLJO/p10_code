local node = torch.class('dqn.Node')


function node:__init(args)

    self.s                             = args.s
    self.idx                           = args.idx
    self.visits                        = 0
    self.directions_to_node            = {}
    self.initial_directions_mag        = {}
    self.distance_scale                = {}

    self.visit_pct                     = 0
    self.visit_pct_debiased            = 0
    self.visit_pct_mom                 = args.visit_pct_mom
    self.num_visit_pct_updates         = 0

    self.actual_visits                 = 0

    -- For ensuring that the visit count is non-decreasing
    self.max_visits                    = 0
end


function node:update_visits(was_visited, episode_count)

    local target = 0
    if was_visited then
        target = 1
        self.actual_visits = self.actual_visits + 1
    end

    self.visit_pct = self.visit_pct_mom * self.visit_pct + (1 - self.visit_pct_mom) * target

    -- Zero debias as in Adam paper
    self.num_visit_pct_updates = self.num_visit_pct_updates + 1
    self.visit_pct_debiased = self.visit_pct / (1 - self.visit_pct_mom ^ self.num_visit_pct_updates)
    self.visits = self.visit_pct_debiased * episode_count

    -- Take max with actual visits so that the count will increase in the long run
    self.visits = math.max(self.visits, self.actual_visits)

    -- Ensure non-decreasing visit count
    if self.visits > self.max_visits then
        self.max_visits = self.visits
    end
    self.visits = math.max(self.visits, self.max_visits)
end


function node:save_img(logfilePath)

    local filename = logfilePath .. "nodes_" .. self.idx .. ".png"
    local img = self.s:clone():reshape(84, 84)
    image.save(filename, img)
end


function node:add_directions_to_node(args)

    local to_node = args.to_node
    local directions = args.directions

    local directions_mag = self:calculate_directions_mag(directions)

    if not self.directions_to_node[to_node] then
        self.initial_directions_mag[to_node] = directions_mag
        self.distance_scale[to_node] = 1
    else
        self.distance_scale[to_node] = directions_mag / self.initial_directions_mag[to_node]
    end

    self.directions_to_node[to_node] = directions
end


function node:calculate_directions_mag(directions)

    -- Use L1 magnitude, as that's what we do in calculate_aux_reward_difference
    local direction_mag = 0
    for i = 1, #directions do
        direction_mag = direction_mag + math.abs(directions[i])
    end
    return direction_mag
end


function node:refresh_directions_to_nodes(args)

    local nodes = args.nodes
    local net = args.net
    local gpu = args.gpu

    if #nodes > 0 then

        local net_input = torch.ByteTensor(#nodes, 2, 84, 84):fill(0)

        if gpu and gpu >= 0 then
            net_input = net_input:cuda()
        else
            net_input = net_input:float()
        end

        for i = 1, #nodes do
            net_input[i][1]:copy(self.s)
            net_input[i][2]:copy(nodes[i].s)
        end

        local net_output = net:forward(net_input):float()

        for i = 1, #nodes do
            local directions = {}
            for j = 1, net_output[i]:size(1) do
                directions[j] = net_output[i][j]
            end
            self:add_directions_to_node{to_node=i, directions=directions}
        end
    end
end


function node:get_direction_diff_to(args)

    local directions_to_current_state = args.directions_to_current_state

    local direction_difference = {}

    for dest_node_idx = 1, #self.directions_to_node do

        direction_difference[dest_node_idx] = calculate_aux_reward_difference{r1=self.directions_to_node[dest_node_idx], r2=directions_to_current_state}

        --direction_difference[dest_node_idx] = calculate_aux_reward_difference{r1=self.directions_to_node[dest_node_idx], r2=directions_to_current_state} / self.distance_scale[dest_node_idx]

        --direction_difference[dest_node_idx] = calculate_cosine_difference{r1=self.directions_to_node[dest_node_idx], r2=directions_to_current_state}
    end

    return direction_difference
end


function node:get_direction_diff_from(args)

    local directions_from_current_state = args.directions_from_current_state

    local direction_difference = {}

    for dest_node_idx = 1, #self.directions_to_node do

        direction_difference[dest_node_idx] = calculate_aux_reward_difference{r1=self.directions_to_node[dest_node_idx], r2=directions_from_current_state[dest_node_idx]}

        --direction_difference[dest_node_idx] = calculate_aux_reward_difference{r1=self.directions_to_node[dest_node_idx], r2=directions_from_current_state[dest_node_idx]} / self.distance_scale[dest_node_idx]

        --direction_difference[dest_node_idx] = calculate_cosine_difference{r1=self.directions_to_node[dest_node_idx], r2=directions_from_current_state[dest_node_idx]}
    end

    return direction_difference
end
