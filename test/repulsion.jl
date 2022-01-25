using Gen
using Ensembles


function rep_test()
    # TODO
    # first iniitalize a scene
    # then simulation for 10 steps
    gm = RepulsionGM()
    init_state = Gen.simulate(rpl_init, (gm,))
    for i = 1:10
    end
end
