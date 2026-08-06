@testitem "COGNAClite" begin
    using Clapeyron

    const COGNAC_SYSTEMS = [
        ["ethanol", "water"] => (;
            smiles=["CCO", "O"],
            T=300.,
            x=[0.2, 0.8],
            lnγ=[0.7587359873715098, 0.06841441733037978],
            idx=["1_1C", "1_2C"],
            R=[1.5540564060211182, 0.9537124037742615],
            Q=[1.8155049085617065, 1.2543470859527588],
            A=25.987350463867188,
        ),
        ["ethanol", "water", "hexane"] => (;
            smiles=["CCO", "O", "CCCCCC"],
            T=300.,
            x=[0.2, 0.4, 0.4],
            lnγ=[0.2608592911460895, 1.5082098352387199, 1.2631358632711551],
            idx=["1_1C", "1_2C"],
            R=[1.5540564060211182, 0.9537124037742615],
            Q=[1.8155049085617065, 1.2543470859527588],
            A=25.987350463867188,
        ),
        ["benzene", "acetone", "ethylamine"] => (;
            smiles=["c1ccccc1", "CC(=O)C", "CCN"],
            T=300.,
            x=[0.2, 0.4, 0.4],
            lnγ=[0.1808951847187626, 0.09209348497616254, 0.046526263884452995],
            idx=["1_1C", "1_2C"],
            R=[0.8897197842597961, 0.8897197842597961],
            Q=[0.9647823572158813, 0.9647823572158813],
            A=0.0,
        ),
        ["water", "acetic acid"] => (;
            smiles=["O", "CC(=O)O"],
            T=300.,
            x=[0.2, 0.8],
            lnγ=[0.261263686400782, 0.01961388287707133],
            idx=["1_1O", "1_2H"],
            R=[0.800090491771698, 0.24720974266529083],
            Q=[0.9672906398773193, 0.3433423936367035],
            A=-842.3084716796875,
        ),
    ]

    for (system, data) in COGNAC_SYSTEMS
        model = COGNAClite(system; userlocations=(; SMILES=data.smiles))

        # Parameter testing
        @test model.params.R[data.idx[1]] ≈ data.R[1]   rtol=1e-6
        @test model.params.R[data.idx[2]] ≈ data.R[2]   rtol=1e-6
        @test model.params.Q[data.idx[1]] ≈ data.Q[1]   rtol=1e-6
        @test model.params.Q[data.idx[2]] ≈ data.Q[2]   rtol=1e-6
        @test model.params.A[data.idx...] ≈ data.A      rtol=1e-6

        # Activity coefficients
        lnγ = log.(activity_coefficient(model, 1e5, data.T, data.x))
        for i in eachindex(system)
            @test lnγ[i] ≈ data.lnγ[i]                  rtol=1e-5
        end
    end
end
