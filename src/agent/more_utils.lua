function dump_to_str(o)

    if type(o) == 'table' then

        local s = '{'

        for k, v in pairs(o) do

            if type(k) ~= 'number' then
                k = '"' .. k.. '"'
            end
            s = s .. '[' .. k .. '] = ' .. dump_to_str(v) .. ', '
        end

        return s .. '} '
    else
        return tostring(o)
    end
end


--[[
function calculate_aux_reward_difference(args)

    local r1 = args.r1
    local r2 = args.r2
    local difference, err1, err2 = 0, 0, 0

    for i = 1, #r1 do
        difference = difference + (r1[i] - r2[i]) * (r1[i] - r2[i])
    end

    return math.sqrt(difference)
end
--]]


function calculate_aux_reward_difference(args)

    local r1 = table_deep_copy(args.r1)
    local r2 = table_deep_copy(args.r2)
    local difference, err1, err2 = 0, 0, 0

    --[[
    for i = 1, #r1 do
        --r1[i] = math.min(0.5, math.max(-0.5, r1[i]))
        --r2[i] = math.min(0.5, math.max(-0.5, r2[i]))
        local alpha = 1
        r1[i] = math.tanh(r1[i] / alpha)
        r2[i] = math.tanh(r2[i] / alpha)
    end
    --]]

    --[[
    for i = 1, #r1 do
        difference = difference + math.abs(r1[i] - r2[i])
        err1 = err1 + r1[i]
        err2 = err2 + r2[i]
    end

    err1 = math.abs(err1)
    err2 = math.abs(err2)
    --]]

    for i = 1, #r1 do
        difference = difference + (r1[i] - r2[i]) * (r1[i] - r2[i])
        err1 = err1 + r1[i]
        err2 = err2 + r2[i]
    end

    --[[
    local limit = 1.5
    if err1 > limit or err2 > limit then
        return -1
    else
        return difference
    end
    --]]

    return difference
end


function calculate_cosine_difference(args)

    local r1 = args.r1
    local r2 = args.r2
    local dot_prod, sqr_mag_1, sqr_mag_2 = 0, 0, 0

    for i = 1, #r1 do
        dot_prod = dot_prod + r1[i] * r2[i]
        sqr_mag_1 = sqr_mag_1 + r1[i] * r1[i]
        sqr_mag_2 = sqr_mag_2 + r2[i] * r2[i]
    end

    local cosine_similarity = dot_prod / math.sqrt(sqr_mag_1 * sqr_mag_2)

    return 1 - cosine_similarity
end


function calculate_kl_divergence(args)

    local bl_distribution = args.bl_distribution
    local new_distribution = args.new_distribution
    local kl_divergence = 0

    for i = 1, #bl_distribution do

        if new_distribution[i] <= 0 then

            if bl_distribution[i] > 0 then
                return 9.999
            else
                -- Both probabilities are zero; treat as zero divergence
            end

        else
            if bl_distribution[i] > 0 then
                kl_divergence = kl_divergence - bl_distribution[i] * math.log(new_distribution[i] / bl_distribution[i])
            else
                -- Do nothing since lim x->0 {x.log(x)} = 0
            end
        end
    end

    return kl_divergence
end


function normalise_to_sum_one(input_arr)

    local result = {}
    local sum = 0

    for i = 1, #input_arr do
        sum = sum + input_arr[i]
    end

    for i = 1, #input_arr do
        result[i] = input_arr[i] / sum
    end

    return result, sum
end

function table_deep_copy(obj, seen)

    if type(obj) ~= 'table' then
        return obj
    end

    if seen and seen[obj] then
        return seen[obj]
    end

    local s = seen or {}
    local res = setmetatable({}, getmetatable(obj))
    s[obj] = res

    for k, v in pairs(obj) do
        res[table_deep_copy(k, s)] = table_deep_copy(v, s)
    end

    return res
end
