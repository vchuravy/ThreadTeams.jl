@testset "nested" begin
    m = zeros(Int, 20, 30)
    inner = zeros(Int, 20)
    tforeach(1:20; scheduler = team) do j
        inner[j] = team_nchunks(1:30)
        tforeach(i -> (m[j, i] = i + 100j), 1:30; scheduler = team)
    end
    @test m == [i + 100j for j in 1:20, i in 1:30]
    @test all(==(1), inner)
    @test tmapreduce(
        j -> tmapreduce(i -> i * j, +, 1:10; scheduler = team), +, 1:10;
        scheduler = team
    ) == 55 * 55
    # OhMyThreads' own schedulers nested in a pool loop are not affected
    @test tmapreduce(j -> tmapreduce(i -> i * j, +, 1:10), +, 1:10; scheduler = team) == 55 * 55
    # and the pool nested in OhMyThreads' schedulers: the first loop to arrive takes the
    # pool, the others run on tasks
    @test tmapreduce(j -> tmapreduce(i -> i * j, +, 1:10; scheduler = team), +, 1:10) == 55 * 55
end
