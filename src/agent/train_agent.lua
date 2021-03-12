--[[
Copyright (c) 2014 Google Inc.

See LICENSE file for full terms of limited license.
]]

if not dqn then
    require "initenv"
end

local cmd = torch.CmdLine()
cmd:text()
cmd:text('Train Agent in Environment:')
cmd:text()
cmd:text('Options:')

cmd:option('-env', '', 'name of environment to use')
cmd:option('-game_path', '', 'path to environment file (ROM)')
cmd:option('-env_params', '', 'string of environment parameters')
cmd:option('-pool_frms', '',
           'string of frame pooling parameters (e.g.: size=2,type="max")')
cmd:option('-actrep', 1, 'how many times to repeat action')
cmd:option('-random_starts', 0, 'play action 0 between 1 and random_starts ' ..
           'number of times at the start of each training episode')

cmd:option('-name', '', 'filename used for saving network and training history')
cmd:option('-network', '', 'reload pretrained network')
cmd:option('-image_compare_network', '', 'reload pretrained exploratory effort network')
cmd:option('-agent', '', 'name of agent file to use')
cmd:option('-agent_params', '', 'string of agent parameters')
cmd:option('-seed', 1, 'fixed input seed for repeatable experiments')
cmd:option('-saveNetworkParams', false, 'saves the agent network in a separate file')
cmd:option('-prog_freq', 5*10^3, 'frequency of progress output')
cmd:option('-save_freq', 5*10^4, 'the model is saved every save_freq steps')
cmd:option('-eval_freq', 10^4, 'frequency of greedy evaluation')
cmd:option('-save_versions', 0, '')

cmd:option('-steps', 10^5, 'number of training steps to perform')
cmd:option('-eval_steps', 10^5, 'number of evaluation steps')

cmd:option('-verbose', 2,
           'the higher the level, the more information is printed to screen')
cmd:option('-threads', 1, 'number of BLAS threads')
cmd:option('-gpu', -1, 'gpu flag')

cmd:text()

local opt = cmd:parse(arg)

--- General setup.
local game_actions, agent, opt = setup(opt)

-- override print to always flush the output
local old_print = print
local print = function(...)
    old_print(...)
    io.flush()
end

local learn_start = agent.learn_start
local start_time = sys.clock()
local reward_counts = {}
local episode_counts = {}
local time_history = {}
local v_history = {}
local qmax_history = {}
local td_history = {}
local reward_history = {}
local step = 0
time_history[1] = 0

local total_reward
local nrewards
local nepisodes

local screen, reward, terminal, all_lives_lost = aleWrap:getState()
local episode_reward = reward
local first_screen = screen:clone()

-- For stats
local score_averaging_episodes = 100

local ep_num = 1
local episode_scores = {}
local episode_scores_idx = 1
local sum_episode_scores = 0
local average_score = 0

local full_episode_score = 0

local full_ep_num = 1
local full_episode_scores = {}
local full_episode_scores_idx = 1
local full_sum_episode_scores = 0
local full_average_score = 0

agent:log("Steps,RealSteps,Episodes,AverageScore,FullEpisodes,AverageFullScore")

print("Iteration ..", step)
while step < opt.steps do
    step = step + 1
    local action_index = agent:perceive(reward, screen, terminal, nil, nil, first_screen)

    if step % 2000 == 0 then
        agent:log(step .. "," .. (step * opt.actrep) .. "," .. ep_num .. "," .. average_score .. "," .. full_ep_num .. "," .. full_average_score)
    end

    -- game over? get next game!
    if not terminal then
        screen, reward, terminal, all_lives_lost = aleWrap:step(game_actions[action_index], true)
        episode_reward = episode_reward + reward
    else

        if episode_scores[episode_scores_idx] then
            sum_episode_scores = sum_episode_scores - episode_scores[episode_scores_idx]
        end

        sum_episode_scores = sum_episode_scores + episode_reward
        episode_scores[episode_scores_idx] = episode_reward
        episode_scores_idx = episode_scores_idx % score_averaging_episodes + 1
        average_score = sum_episode_scores / #episode_scores

        full_episode_score = full_episode_score + episode_reward

        print("End of episode " .. ep_num .. ", total score: " .. episode_reward .. ", average score = " .. average_score)

        ep_num = ep_num + 1

        if all_lives_lost then

            if full_episode_scores[full_episode_scores_idx] then
                full_sum_episode_scores = full_sum_episode_scores - full_episode_scores[full_episode_scores_idx]
            end

            full_sum_episode_scores = full_sum_episode_scores + full_episode_score
            full_episode_scores[full_episode_scores_idx] = full_episode_score
            full_episode_scores_idx = full_episode_scores_idx % score_averaging_episodes + 1
            full_average_score = full_sum_episode_scores / #full_episode_scores

            print("***End of full episode " .. full_ep_num .. ", total score: " .. full_episode_score .. ", average score = " .. full_average_score)
            print("")

            full_ep_num = full_ep_num + 1

            full_episode_score = 0
        end

        if opt.random_starts > 0 then
            screen, reward, terminal, all_lives_lost = aleWrap:nextRandomGame()
        else
            screen, reward, terminal, all_lives_lost = aleWrap:newGame()
        end

        episode_reward = reward
    end

    if step % opt.prog_freq == 0 then
        assert(step==agent.numSteps, 'trainer step: ' .. step ..
                ' & agent.numSteps: ' .. agent.numSteps)
        print("Steps: ", step)
        agent:report()
        collectgarbage()
    end

    if step%1000 == 0 then collectgarbage() end

    if step % opt.eval_freq == 0 and step > learn_start then

        screen, reward, terminal, all_lives_lost = aleWrap:newGame()

        total_reward = 0
        nrewards = 0
        nepisodes = 0
        episode_reward = 0

        local eval_time = sys.clock()
        for estep=1,opt.eval_steps do
            local action_index = agent:perceive(reward, screen, terminal, true, 0.05)

            -- Play game in test mode (episodes don't end when losing a life)
            screen, reward, terminal, all_lives_lost = aleWrap:step(game_actions[action_index])

            if estep%1000 == 0 then collectgarbage() end

            -- record every reward
            episode_reward = episode_reward + reward
            if reward ~= 0 then
               nrewards = nrewards + 1
            end

            if terminal then
                total_reward = total_reward + episode_reward
                episode_reward = 0
                nepisodes = nepisodes + 1
                screen, reward, terminal, all_lives_lost = aleWrap:nextRandomGame()
            end
        end

        eval_time = sys.clock() - eval_time
        start_time = start_time + eval_time
        agent:compute_validation_statistics()
        local ind = #reward_history+1
        total_reward = total_reward/math.max(1, nepisodes)

        if #reward_history == 0 or total_reward > torch.Tensor(reward_history):max() then
            agent.best_network = agent.network:clone()
        end

        if agent.v_avg then
            v_history[ind] = agent.v_avg
            td_history[ind] = agent.tderr_avg
            qmax_history[ind] = agent.q_max
        end
        print("V", v_history[ind], "TD error", td_history[ind], "Qmax", qmax_history[ind])

        reward_history[ind] = total_reward
        reward_counts[ind] = nrewards
        episode_counts[ind] = nepisodes

        time_history[ind+1] = sys.clock() - start_time

        local time_dif = time_history[ind+1] - time_history[ind]

        local training_rate = opt.actrep*opt.eval_freq/time_dif

        print(string.format(
            '\nSteps: %d (frames: %d), reward: %.2f, epsilon: %.2f, lr: %G, ' ..
            'training time: %ds, training rate: %dfps, testing time: %ds, ' ..
            'testing rate: %dfps,  num. ep.: %d,  num. rewards: %d',
            step, step*opt.actrep, total_reward, agent.ep, agent.lr, time_dif,
            training_rate, eval_time, opt.actrep*opt.eval_steps/eval_time,
            nepisodes, nrewards))
    end

    if step % opt.save_freq == 0 or step == opt.steps then

        -- Back up values before setting to nil
        local s, a, r, s2, term = agent.valid_s, agent.valid_a, agent.valid_r,
            agent.valid_s2, agent.valid_term

        agent.valid_s, agent.valid_a, agent.valid_r, agent.valid_s2,
            agent.valid_term = nil, nil, nil, nil, nil, nil, nil

        -- Now save everything
        local filename = opt.name

        if opt.save_versions > 0 then
            filename = filename .. "_" .. math.floor(step / opt.save_versions)
        end

        torch.save(filename .. "_q_net.t7", agent.network:clearState())
        torch.save(filename .. "_dist_net.t7", agent.image_compare_network:clearState())

        if opt.saveNetworkParams then
            local nets = {network=w:clone():float()}
            torch.save(filename..'.params.t7', nets, 'ascii')
            local image_compare_nets = {image_compare_network=image_compare_w:clone():float()}
            torch.save(filename..'.image_compare_params.t7', image_compare_nets, 'ascii')
        end

        print('Saved:', filename .. '.t7')
        io.flush()
        collectgarbage()
        collectgarbage()
    end
end
