previous_combo = 0

function hits_and_time()
    if data.combo > previous_combo then
        previous_combo = data.combo
        return 10
    else
        if data.combo > 0 then
            return 1
        else
            return 0
        end
    end
end