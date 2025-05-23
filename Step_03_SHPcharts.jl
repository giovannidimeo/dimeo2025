# Load packages
begin
    using Pkg
    using BenchmarkTools
    using DataFrames
    using DataFramesMeta
    using Interpolations
    using Plots
    using XLSX
    using Distributions
    using Combinatorics
    using DataInterpolations
    using JLD
    using Roots
    using Random
    using Interpolations
    using CSV
    using StatsBase
    using Measures
end

begin # Set directories 
    background_data = raw"~\background_data"
    results = raw"~\results"
    figures = raw"~\figures"
end 


function winsor2(var, prop)
    per = percentile.(eachcol(var), prop)
    var = ifelse.(var .> per, per, var)
    return(var)
end

function data_prep(df)
    # Generate an id by grouping by `time` and using row number within each group
    transform!(groupby(df, [:time]), :time => eachindex => :id)

    # Pull id to the fron 
    select!(df, [:id, :time], :)

    # Sort by id and time
    sort!(df, [:id, :time])

    # Create flag for (eventual) owner
    transform!(groupby(df, [:id]), :DT => maximum => :owner)

    # Create variable for total wealth 
    df.purchase = ifelse.(df.DP .== 1, df.time, 0)
    transform!(groupby(df, [:id]), :purchase => maximum => :purchase)

    # Sort by id and time
    sort!(df, [:id, :time])

    return(df)
end 

function winsor_down(var, prop)
    per = percentile.(eachcol(var), prop)
    var = ifelse.(var .< per, per, var)
    return(var)
end

######################################################################
# LOAD DATA

begin # load data
    cd(results)
    
    df_or = CSV.read("df_d.csv", DataFrame)  
    print("done")
end 

df_main = data_prep(df_or)

######################################################################
### COMPARISON WITH SHP
default(;fontfamily="serif-roman")
cd(background_data)
index_df = CSV.read("index.csv", DataFrame)
index_df.time .= index_df.year .- 1977
select!(index_df, [:time, :year, :index_forecast])
leftjoin!(df_main, index_df, on = :time) 
df_main.tot_wealth = df_main.ST .+ df_main.DT .* df_main.HT .* df_main.pt .* df_main.OT
df_main.property = df_main.DT .* df_main.HT .* df_main.pt .* df_main.OT

df_main.ST .= df_main.ST ./ df_main.index_forecast ./ 10
df_main.tot_wealth .= df_main.tot_wealth ./ df_main.index_forecast ./ 10
df_main.YT .= df_main.YT ./ df_main.index_forecast ./ 10
df_main.property = df_main.property ./ df_main.index_forecast ./ 10

df_main.ST = winsor2(df_main.ST, 95)
df_main.tot_wealth = winsor2(df_main.tot_wealth, 95)
df_main.property = winsor2(df_main.property, 95)

df_main.data  .= "simul"

df_main.age .= df_main.time .+ 24

df_main.h_costs = df_main.HT .* df_main.pt .* (ψ_r + 0.2) .* df_main.DP .* df_main.DT + df_main.HT .* df_main.pt .* (1 .- df_main.OT) .* (ψ_r) .* (1 .- df_main.DP) .* df_main.DT + df_main.HT .* df_main.pt .* (ψ_r) .* (1 .- df_main.DT)

df_main.tot_costs = (df_main.h_costs) ./ df_main.index_forecast ./ 10
df_main.tot_costs = winsor2(df_main.tot_costs, 95)

df_main = select(@rsubset(df_main, :time <= 70), [:id, :time, :age, :owner, :ST, :YT, :DT, :property, :tot_wealth, :tot_costs])

df_mean = combine(groupby(df_main, [:age]), [:ST, :YT, :property, :tot_wealth, :tot_costs, :DT] .=> mean, renamecols = false)
df_mean.data  .= "simul"

df_mean_owner = combine(groupby(@rsubset(df_main, :DT ==1), [:age]), [:ST, :YT, :property, :tot_wealth, :tot_costs, :DT] .=> mean, renamecols = false)
df_mean_owner.data  .= "simul"

cd(background_data)
shp = CSV.read("shp.csv", DataFrame)
shp_owner = CSV.read("shp_owner.csv", DataFrame)

df_mean = vcat(df_mean, shp)
df_mean_owner = vcat(df_mean_owner, shp)

sort!(df_mean, [:data, :age])
sort!(df_mean_owner, [:data, :age])

plot(@rsubset(df_mean, :data == "simul").age, 
@rsubset(df_mean, :data == "simul").YT,
title = "A: Available Income",
xlabel = "Age",
ylabel = "100,000 CHF",
linecolor = "red",
label="Simulation",
xticks = 25:10:95,
ylimits = (0,1),
legend = :topleft,
legend_font_pointsize = 11)

plot!(@rsubset(df_mean, :data == "shp").age, 
@rsubset(df_mean, :data == "shp").YT,
linecolor = "navy",
label="SHP")

p1 =plot!()

plot(@rsubset(df_mean, :data == "simul").age, 
@rsubset(df_mean, :data == "simul").tot_costs,
title = "B: Housing Expenditures",
xlabel = "Age",
ylabel = "100,000 CHF",
linecolor = "red",
label="Simulation",
xticks = 25:10:95,
ylimits = (0,0.25),
legend = :topleft,
legend_font_pointsize = 11)

plot!(@rsubset(df_mean, :data == "shp").age, 
@rsubset(df_mean, :data == "shp").tot_costs,
linecolor = "navy",
label="SHP")

p2 =plot!()

plot(@rsubset(df_mean, :data == "simul").age, 
@rsubset(df_mean, :data == "simul").ST,
title = "C: Liquid Wealth",
xlabel = "Age",
ylabel = "100,000 CHF",
linecolor = "red",
label="Simulation",
xticks = 25:10:95,
ylimits = (0,5),
legend = :topleft,
legend_font_pointsize = 11)

plot!(@rsubset(df_mean, :data == "shp").age, 
@rsubset(df_mean, :data == "shp").ST,
linecolor = "navy",
label="SHP")

p3 =plot!()

plot(@rsubset(df_mean, :data == "simul").age, 
@rsubset(df_mean, :data == "simul").DT,
title = "D: Share of Homeowners",
xlabel = "Age",
ylabel = "Share",
linecolor = "red",
label="Simulation",
xticks = 25:10:95,
ylimits = (0,1),
legend = :topleft,
legend_font_pointsize = 11)

plot!(@rsubset(df_mean, :data == "shp").age, 
@rsubset(df_mean, :data == "shp").DT,
linecolor = "navy",
label="SHP")

p4 =plot!()


plot(@rsubset(df_mean_owner, :data == "simul").age, 
@rsubset(df_mean_owner, :data == "simul").property,
title = "E: Net Value of Property",
xlabel = "Age",
ylabel = "100,000 CHF",
linecolor = "red",
label="Simulation",
xticks = 25:10:95,
ylimits = (0,5),
legend = :topleft,
legend_font_pointsize = 11)

plot!(@rsubset(df_mean_owner, :data == "shp").age, 
@rsubset(df_mean_owner, :data == "shp").property,
linecolor = "navy",
label="SHP")

p5 = plot!()

plot(@rsubset(df_mean, :data == "simul").age, 
@rsubset(df_mean, :data == "simul").tot_wealth,
title = "F: Total Net Wealth",
xlabel = "Age",
ylabel = "100,000 CHF",
linecolor = "red",
label="Simulation",
xticks = 25:10:95,
ylimits = (0,10),
legend = :topleft,
legend_font_pointsize = 11)

plot!(@rsubset(df_mean, :data == "shp").age, 
@rsubset(df_mean, :data == "shp").tot_wealth,
linecolor = "navy",
label="SHP")

p6 =plot!()

plot(p1, p2, p3, p4, p5, p6, layout = (3, 2))

plot!(size=(1000,1500), left_margin=7mm, 
    bottom_margin = 8mm, top_margin = 7mm) 
cd(figures)
savefig(raw"simul_shp.pdf")

