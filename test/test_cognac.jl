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

@testitem "COGNACexp/COGNACimp" begin
    using Clapeyron

    const COGNACEXP_SYSTEMS = [
        ["ethanol", "water"] => (;
            smiles=["CCO", "O"],
            T=300.0,
            x=[0.2, 0.8],
            lnγ=[0.7539994444662604, 0.0722208074171912],
            idx=["1_1C", "2_7H"],
            R=[1.4782928228378296, 0.2291431725025177],
            Q=[1.7634624242782593, 0.3947049081325531],
            A=-538.0289611816406,
            B=3.060164962581754,
            C=-0.004847234541637839,
        ),
        ["ethanol", "water", "hexane"] => (;
            smiles=["CCO", "O", "CCCCCC"],
            T=300.0,
            x=[0.2, 0.4, 0.4],
            lnγ=[0.36078927099053465, 1.480575238504177, 1.145342121589834],
            idx=["1_1C", "3_13C"],
            R=[1.4782928228378296, 1.5042107105255127],
            Q=[1.7634626626968384, 1.806667447090149],
            A=5.2985382080078125,
            B=-0.013374509016539495,
            C=1.3723627139531894e-06,
        ),
        ["benzene", "acetone", "ethylamine"] => (;
            smiles=["c1ccccc1", "CC(=O)C", "CCN"],
            T=300.0,
            x=[0.2, 0.4, 0.4],
            lnγ=[0.24773115201533963, 0.04957852632785954, 0.05972526448252066],
            idx=["1_1C", "3_15H"],
            R=[0.9022020697593689, 0.3000721037387848],
            Q=[0.9606481790542603, 0.3686860203742981],
            A=-40.917999267578125,
            B=0.10985980113040836,
            C=-0.001151896377401141,
        ),
        ["water", "acetic acid"] => (;
            smiles=["O", "CC(=O)O"],
            T=300.0,
            x=[0.2, 0.8],
            lnγ=[0.3085035719319726, 0.0330090302051379],
            idx=["1_1O", "2_8H"],
            R=[0.8262559771537781, 0.2526954412460327],
            Q=[0.991247296333313, 0.38389480113983154],
            A=-762.4626770019531,
            B=4.6732843256068675,
            C=-0.007991655160012332,
        ),
    ]

    const COGNACIMP_SYSTEMS = [
        ["ethanol", "water"] => (;
            smiles=["CCO", "O"],
            T=300.0,
            x=[0.2, 0.8],
            lnγ=[0.760787784168858, 0.0704419179003873],
            idx=["1_1C", "2_7H"],
            R=[1.6503764390945435, 0.10758184641599655],
            Q=[1.8456826210021973, 0.42281481623649597],
            A=-1416.827163696289,
            B=7.365597371944701,
            C=-0.009617966942643452,
        ),
        ["ethanol", "water", "hexane"] => (;
            smiles=["CCO", "O", "CCCCCC"],
            T=300.0,
            x=[0.2, 0.4, 0.4],
            lnγ=[0.43262891033312806, 1.3631324465334471, 0.9840875029552478],
            idx=["1_1C", "3_13C"],
            R=[1.6503762006759644, 1.6296448707580566],
            Q=[1.8456823825836182, 1.7548329830169678],
            A=-10.37200927734375,
            B=0.21139351849425625,
            C=-0.00045406650183397996,
        ),
        ["benzene", "acetone", "ethylamine"] => (;
            smiles=["c1ccccc1", "CC(=O)C", "CCN"],
            T=300.0,
            x=[0.2, 0.4, 0.4],
            lnγ=[0.28100429407272953, -0.08366940068398962, -0.10603712976279005],
            idx=["1_1C", "3_15H"],
            R=[0.9090990424156189, 0.16710464656352997],
            Q=[0.9585369229316711, 0.37491920590400696],
            A=-3.8218994140625,
            B=-1.486913440172732,
            C=-0.000743185819565981,
        ),
        ["water", "acetic acid"] => (;
            smiles=["O", "CC(=O)O"],
            T=300.0,
            x=[0.2, 0.8],
            lnγ=[0.2697908996004456, 0.03869592682561781],
            idx=["1_1O", "2_8H"],
            R=[1.1172096729278564, 0.12031781673431396],
            Q=[1.369924545288086, 0.39198794960975647],
            A=-1601.203125,
            B=7.749546683716251,
            C=-0.012322326539929039,
        ),
    ]

    for (Model, systems) in ((COGNACexp, COGNACEXP_SYSTEMS), (COGNACimp, COGNACIMP_SYSTEMS))
        for (system, data) in systems
            model = Model(system; userlocations=(; SMILES=data.smiles))

            # Parameter testing
            @test model.params.R[data.idx[1]] ≈ data.R[1]   rtol=1e-5
            @test model.params.R[data.idx[2]] ≈ data.R[2]   rtol=1e-5
            @test model.params.Q[data.idx[1]] ≈ data.Q[1]   rtol=1e-5
            @test model.params.Q[data.idx[2]] ≈ data.Q[2]   rtol=1e-5
            # looser: the exchange subtraction cancels two Float32 numbers of the same size
            @test model.params.A[data.idx...] ≈ data.A      rtol=1e-3
            @test model.params.B[data.idx...] ≈ data.B      rtol=1e-3
            @test model.params.C[data.idx...] ≈ data.C      rtol=1e-3

            # Ψ_kk = 1 at every temperature
            for k in data.idx, P in (model.params.A, model.params.B, model.params.C)
                @test iszero(P[k, k])
            end

            # Activity coefficients
            lnγ = log.(activity_coefficient(model, 1e5, data.T, data.x))
            for i in eachindex(system)
                @test lnγ[i] ≈ data.lnγ[i]                  rtol=1e-5
            end

            # lnγ → 0 for a pure component (up to the implicit solver's truncation)
            x1 = [i == 1 ? 1.0 : 0.0 for i in eachindex(system)]
            @test log(activity_coefficient(model, 1e5, data.T, x1)[1]) ≈ 0.0 atol=1e-6
        end
    end
end
