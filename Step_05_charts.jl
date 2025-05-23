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
    using StatsPlots
    using RData
    default(;fontfamily="serif-roman")
end

begin # Set directories 
    background_data = raw"~\background_data"
    results = raw"~\results"
    figures = raw"~\figures"
end 

######################################################################
# CREATE DATA

begin # load data
    cd(results)
    
    df = CSV.read("dataready.csv", DataFrame)  
    print("done")

    rresults = load("AllResults.RData")
    print("done")

    for i = 1:7
        replace!(rresults["MainResults_Megaloop"][i].upper, missing => 0)
        replace!(rresults["MainResults_Megaloop"][i].lower, missing => 0)
    end
end 

######################################################################
### Participation costs 
leftjoin!(df, combine(groupby(df, [:time]), [:YT] .=> mean), on = :time) 

df.at_costs = df.AT .* df.YT_mean .* 0.01
describe(@rsubset(df, :AP == 1).at_costs)

### Downpayment
df.downpayment .=  df.HT .* df.pt .* (0.2) .* df.DP .* df.DT .* (1 .- df.counterfactual)
describe(@rsubset(df, :DP == 1, :counterfactual == 0 ).downpayment)

### Averages
mean(@rsubset(df, :time >= 45-25, :time <= 65-25, :owner == 1, :counterfactual == 0).ST)/
mean(@rsubset(df, :time >= 45-25, :time <= 65-25, :owner == 1, :counterfactual == 1).ST) -1

mean(@rsubset(df, :time == 65 - 25, :owner == 1, :counterfactual == 0).ST)/
mean(@rsubset(df, :time == 65 - 25, :owner == 1, :counterfactual == 1).ST) -1

mean(@rsubset(df, :time >= 65-25, :time <= 95-25, :owner == 1, :counterfactual == 0).ST)/
mean(@rsubset(df, :time >= 65-25, :time <= 95-25, :owner == 1, :counterfactual == 1).ST) -1

mean(@rsubset(df, :time >= 80-25, :time <= 95-25, :owner == 1, :counterfactual == 0).tot_wealth)/
mean(@rsubset(df, :time >= 80-25, :time <= 95-25, :owner == 1, :counterfactual == 1).tot_wealth) -1

######################################################################
### Evolution of wealth (section 4)

#############
## figure 2
function collapsed(var) # Collapse dataframe by age
    var_sym = Symbol(var)
    df_mean = combine(groupby(df, [:time, :owner, :counterfactual]), var_sym .=> mean .=> "avg", var_sym .=> std .=> :se, nrow => :count, renamecols = false)
    df_mean = @rsubset(df_mean, :time <= 70)
    
    df_mean.hi = df_mean.avg .+ 1.96 .* df_mean.se ./ sqrt.(df_mean.count)
    df_mean.lo = df_mean.avg .- 1.96 .* df_mean.se ./ sqrt.(df_mean.count)

    return(df_mean)
end

df_mean = collapsed("ST")
df_mean.age .= df_mean.time .+ 24

plot(@rsubset(df_mean, :owner == 1, :counterfactual == 0).age, 
@rsubset(df_mean, :owner == 1, :counterfactual == 0).avg,
ribbon = (@rsubset(df_mean, :owner == 1, :counterfactual == 0).hi .- @rsubset(df_mean, :owner == 1, :counterfactual == 0).avg, 
@rsubset(df_mean, :owner == 1, :counterfactual == 0).avg .- @rsubset(df_mean, :owner == 1, :counterfactual == 0).lo),
title = "A: Liquid Wealth",
xlabel = "Age",
ylabel = "100,000 CHF",
linecolor = "black",
label="Owners",
xticks = 25:10:95,
ylimits = (0,10),
legend = :topleft,
legend_font_pointsize = 11)

plot!(@rsubset(df_mean, :owner == 1, :counterfactual == 1).age, 
@rsubset(df_mean, :owner == 1, :counterfactual == 1).avg,
ribbon = (@rsubset(df_mean, :owner == 1, :counterfactual == 1).hi .- @rsubset(df_mean, :owner == 1, :counterfactual == 1).avg, 
@rsubset(df_mean, :owner == 1, :counterfactual == 1).avg .- @rsubset(df_mean, :owner == 1, :counterfactual == 1).lo),
linecolor = "black",
label="Counterfactual owners",
linestyle = :dashdot)

plot!(@rsubset(df_mean, :owner == 0, :counterfactual == 0).age, 
@rsubset(df_mean, :owner == 0, :counterfactual == 0).avg,
ribbon = (@rsubset(df_mean, :owner == 0, :counterfactual == 0).hi .- @rsubset(df_mean, :owner == 0, :counterfactual == 0).avg, 
@rsubset(df_mean, :owner == 0, :counterfactual == 0).avg .- @rsubset(df_mean, :owner == 0, :counterfactual == 0).lo),
linecolor = "black",
label="Tenants",
linestyle = :dot)

p1 = plot!()

df_mean = collapsed("tot_wealth")
df_mean.age .= df_mean.time .+ 24

plot(@rsubset(df_mean, :owner == 1, :counterfactual == 0).age, 
@rsubset(df_mean, :owner == 1, :counterfactual == 0).avg,
ribbon = (@rsubset(df_mean, :owner == 1, :counterfactual == 0).hi .- @rsubset(df_mean, :owner == 1, :counterfactual == 0).avg, 
@rsubset(df_mean, :owner == 1, :counterfactual == 0).avg .- @rsubset(df_mean, :owner == 1, :counterfactual == 0).lo),
title = "B: Total Net Wealth",
xlabel = "Age",
ylabel = "100,000 CHF",
linecolor = "black",
label="Owners",
xticks = 25:10:95,
ylimits = (0,10),
legend = :topleft,
legend_font_pointsize = 11)

plot!(@rsubset(df_mean, :owner == 1, :counterfactual == 1).age, 
@rsubset(df_mean, :owner == 1, :counterfactual == 1).avg,
ribbon = (@rsubset(df_mean, :owner == 1, :counterfactual == 1).hi .- @rsubset(df_mean, :owner == 1, :counterfactual == 1).avg, 
@rsubset(df_mean, :owner == 1, :counterfactual == 1).avg .- @rsubset(df_mean, :owner == 1, :counterfactual == 1).lo),
linecolor = "black",
label="Counterfactual owners",
linestyle = :dashdot)

plot!(@rsubset(df_mean, :owner == 0, :counterfactual == 0).age, 
@rsubset(df_mean, :owner == 0, :counterfactual == 0).avg,
ribbon = (@rsubset(df_mean, :owner == 1, :counterfactual == 0).hi .- @rsubset(df_mean, :owner == 1, :counterfactual == 0).avg, 
@rsubset(df_mean, :owner == 1, :counterfactual == 0).avg .- @rsubset(df_mean, :owner == 1, :counterfactual == 0).lo),
linecolor = "black",
label="Tenants",
linestyle = :dot)

p2 = plot!()

plot(p1, p2, layour = (1,1))
plot!(size=(1000,500), left_margin=7mm, 
    bottom_margin = 8mm, top_margin = 7mm) 
cd(figures)
savefig("lifecycle_wealth.pdf")


#############
# figure 3

df_owners = df

df_owners.YT_disc = df_owners.YT ./ (1.018 .^ df_owners.time)
transform!(groupby(df_owners, [:id]), :YT_disc => sum => :lifeincome)

df_owners =  @rsubset(df_owners, :time == 70)

q1 = quantile(df_owners.lifeincome, [0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9])[1]
q2 = quantile(df_owners.lifeincome, [0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9])[2]
q3 = quantile(df_owners.lifeincome, [0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9])[3]
q4 = quantile(df_owners.lifeincome, [0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9])[4]
q5 = quantile(df_owners.lifeincome, [0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9])[5]
q6 = quantile(df_owners.lifeincome, [0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9])[6]
q7 = quantile(df_owners.lifeincome, [0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9])[7]
q8 = quantile(df_owners.lifeincome, [0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9])[8]
q9 = quantile(df_owners.lifeincome, [0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9])[9]

df_owners.incomeq .= 0
df_owners.incomeq .= ifelse.((df_owners.lifeincome .<= q1), 1, 0)
df_owners.incomeq .= ifelse.((df_owners.lifeincome .> q1 .&& df_owners.lifeincome .<= q2), 2, df_owners.incomeq )
df_owners.incomeq .= ifelse.((df_owners.lifeincome .> q2 .&& df_owners.lifeincome .<= q3), 3, df_owners.incomeq )
df_owners.incomeq .= ifelse.((df_owners.lifeincome .> q3 .&& df_owners.lifeincome .<= q4), 4, df_owners.incomeq )
df_owners.incomeq .= ifelse.((df_owners.lifeincome .> q4 .&& df_owners.lifeincome .<= q5), 5, df_owners.incomeq )
df_owners.incomeq .= ifelse.((df_owners.lifeincome .> q5 .&& df_owners.lifeincome .<= q6), 6, df_owners.incomeq )
df_owners.incomeq .= ifelse.((df_owners.lifeincome .> q6 .&& df_owners.lifeincome .<= q7), 7, df_owners.incomeq )
df_owners.incomeq .= ifelse.((df_owners.lifeincome .> q7 .&& df_owners.lifeincome .<= q8), 8, df_owners.incomeq )
df_owners.incomeq .= ifelse.((df_owners.lifeincome .> q8 .&& df_owners.lifeincome .<= q9), 9, df_owners.incomeq )
df_owners.incomeq .= ifelse.((df_owners.lifeincome .> q9), 10, df_owners.incomeq)

df_owners = @rsubset(df_owners, :owner == 1)

df_mean = combine(groupby(df_owners, [:incomeq, :counterfactual]), 
    :tot_wealth => mean => :tot_wealth, 
    :tot_wealth => (x -> std(x)/sqrt(length(x))) => :se_wealth)

default(;fontfamily="serif-roman")
p1 = groupedbar(df_mean.incomeq, df_mean.tot_wealth, group = df_mean.counterfactual,  
            yerror = df_mean.se_wealth,
            color = ["#80CDFD" "#F1B7A3"], 
            legend=:topleft,
            label = ["Owners" "Counterfactual owners"],
            xlabel = "Life-income deciles", ylabel = "100,000 CHF",
            title = "A: Total Net Wealth at Age 95", bar_width = 0.67,
            yticks = (0:2:14),
            ylimits = (0,14),
            xticks = (2:1:10, ["20", "30", "40", "50", "60", "70", "80", "90", ">90"]),
            lw = 0,
            grid = :on)


df_mean2 = df_owners[:, [:id, :incomeq, :tot_wealth, :counterfactual]]
df_mean2.id .= ifelse.(df_mean2.counterfactual .== 1 , df_mean2.id .- 10000, df_mean2.id )
df_mean2 = unstack(df_mean2, :id, :counterfactual, :tot_wealth , renamecols=x->Symbol("group", x))
df_mean2 = innerjoin(df_mean2, df_owners[:, [:id, :incomeq]], on = :id)


df_mean2.rel_diff .= df_mean2.group0 ./ df_mean2.group1 .- 1
df_mean2.rel_diff  = winsor2(df_mean2.rel_diff , 99) # there are a few outliers 
df_mean2 = combine(groupby(df_mean2, [:incomeq]), 
            :group0 => mean => :meanown, 
            :group1 => mean => :meancf, 
            :rel_diff => mean => :rel_diff, 
            :rel_diff => (x -> std(x)/sqrt(length(x))) => :se_rel_diff)

plt = twinx()
plot!(plt, df_mean2.incomeq, df_mean2.rel_diff,
yerror=df_mean2.se_rel_diff,
color = :red,
yticks = (-0.2:0.2:1.2),
ylimits = (-0.2,1.2),
ylabel = "Ratio",
legend = false,
grid = :on)

df_owners = df

df_owners.YT_disc = df_owners.YT ./ (1.018 .^ df_owners.time)
transform!(groupby(df_owners, [:id]), :YT_disc => sum => :lifeincome)

q1 = quantile(df_owners.lifeincome, [0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9])[1]
q2 = quantile(df_owners.lifeincome, [0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9])[2]
q3 = quantile(df_owners.lifeincome, [0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9])[3]
q4 = quantile(df_owners.lifeincome, [0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9])[4]
q5 = quantile(df_owners.lifeincome, [0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9])[5]
q6 = quantile(df_owners.lifeincome, [0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9])[6]
q7 = quantile(df_owners.lifeincome, [0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9])[7]
q8 = quantile(df_owners.lifeincome, [0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9])[8]
q9 = quantile(df_owners.lifeincome, [0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9])[9]

df_owners.incomeq .= 0
df_owners.incomeq .= ifelse.((df_owners.lifeincome .<= q1), 1, 0)
df_owners.incomeq .= ifelse.((df_owners.lifeincome .> q1 .&& df_owners.lifeincome .<= q2), 2, df_owners.incomeq )
df_owners.incomeq .= ifelse.((df_owners.lifeincome .> q2 .&& df_owners.lifeincome .<= q3), 3, df_owners.incomeq )
df_owners.incomeq .= ifelse.((df_owners.lifeincome .> q3 .&& df_owners.lifeincome .<= q4), 4, df_owners.incomeq )
df_owners.incomeq .= ifelse.((df_owners.lifeincome .> q4 .&& df_owners.lifeincome .<= q5), 5, df_owners.incomeq )
df_owners.incomeq .= ifelse.((df_owners.lifeincome .> q5 .&& df_owners.lifeincome .<= q6), 6, df_owners.incomeq )
df_owners.incomeq .= ifelse.((df_owners.lifeincome .> q6 .&& df_owners.lifeincome .<= q7), 7, df_owners.incomeq )
df_owners.incomeq .= ifelse.((df_owners.lifeincome .> q7 .&& df_owners.lifeincome .<= q8), 8, df_owners.incomeq )
df_owners.incomeq .= ifelse.((df_owners.lifeincome .> q8 .&& df_owners.lifeincome .<= q9), 9, df_owners.incomeq )
df_owners.incomeq .= ifelse.((df_owners.lifeincome .> q9), 10, df_owners.incomeq)

df_mean = combine(groupby(df_owners, [:id]), 
    :tot_wealth => mean => :tot_wealth)

df_owners = @rsubset(df_owners, :owner == 1)

df_mean = innerjoin(df_mean, @rsubset(df_owners, :time == 1)[:,[:id, :incomeq, :counterfactual]], on = :id)

df_mean_reduced = combine(groupby(df_mean, [:incomeq, :counterfactual]), 
    :tot_wealth => mean => :tot_wealth, 
    :tot_wealth => (x -> std(x)/sqrt(length(x))) => :se_wealth)

p2 = groupedbar(df_mean_reduced.incomeq, df_mean_reduced.tot_wealth, group = df_mean_reduced.counterfactual,  
yerror = df_mean_reduced.se_wealth,
color = ["#80CDFD" "#F1B7A3"],
label = ["Owners" "Counterfactual owners" "Tenants"],
legend=:topleft,
xlabel = "Life-income deciles", ylabel = "100,000 CHF",
title = "B: Average Total Net Wealth", bar_width = 0.67,
yticks = (0:2:14),
ylimits = (0,14),
xticks = (2:1:10, ["20", "30", "40", "50", "60", "70", "80", "90", ">90"]),
lw = 0,
grid = :on)


df_mean2 = df_mean[:, [:id, :incomeq, :tot_wealth, :counterfactual]]
df_mean2.id .= ifelse.(df_mean2.counterfactual .== 1 , df_mean2.id .- 10000, df_mean2.id )
df_mean2 = unstack(df_mean2, :id, :counterfactual, :tot_wealth , renamecols=x->Symbol("group", x))
df_mean2 = innerjoin(df_mean2, df_mean[:, [:id, :incomeq]], on = :id)


df_mean2.rel_diff .= df_mean2.group0 ./ df_mean2.group1 .- 1
df_mean2.rel_diff  = winsor2(df_mean2.rel_diff , 99) # there is one outlier 
df_mean2 = combine(groupby(df_mean2, [:incomeq]), 
            :group0 => mean => :meanown, 
            :group1 => mean => :meancf, 
            :rel_diff => mean => :rel_diff, 
            :rel_diff => (x -> std(x)/sqrt(length(x))) => :se_rel_diff)

plt = twinx()
plot!(plt, df_mean2.incomeq, df_mean2.rel_diff,
yerror=df_mean2.se_rel_diff,
color = :red,
yticks = (-0.2:0.2:1.2),
ylimits = (-0.2,1.2),
ylabel = "Ratio",
legend = false,
grid = :on)


plot(p1, p2, layour = (1,1))

plot!(size=(1000,500), left_margin=7mm, 
    bottom_margin = 8mm, top_margin = 7mm,
    right_margin = 7mm) 
cd(figures)
savefig("income_dec.pdf")


#############
# figure 4
function collapsed(var) # Collapse by event time
    var_sym = Symbol(var)
    df_owners = @rsubset(df, :owner == 1, :purchase > 1)
    df_owners.event .= df_owners.time .- df_owners.purchase
    df_mean = combine(groupby(df_owners, [:event, :counterfactual]), var_sym .=> mean .=> "avg", var_sym .=> std .=> :se, nrow => :count, renamecols = false)
    df_mean = @rsubset(df_mean, :event >= -5, :event <= 30)
    
    df_mean.hi = df_mean.avg .+ 1.96 .* df_mean.se ./ sqrt.(df_mean.count)
    df_mean.lo = df_mean.avg .- 1.96 .* df_mean.se ./ sqrt.(df_mean.count)

    return(df_mean)
end

df_mean = collapsed("ST")

plot(@rsubset(df_mean, :counterfactual == 0).event, 
@rsubset(df_mean,  :counterfactual == 0).avg,
ribbon = (@rsubset(df_mean, :counterfactual == 0).hi .- @rsubset(df_mean, :counterfactual == 0).avg, 
@rsubset(df_mean, :counterfactual == 0).avg .- @rsubset(df_mean, :counterfactual == 0).lo),
title = "A: Liquid Wealth",
xlabel = "Years from home purchase",
ylabel = "100,000 CHF",
linecolor = "black",
label="Owners",
xticks = -5:5:30,
ylimits = (0,10),
legend = :bottomright,
legend_font_pointsize = 11, )

plot!(@rsubset(df_mean, :counterfactual == 1).event, 
@rsubset(df_mean, :counterfactual == 1).avg,
ribbon = (@rsubset(df_mean, :counterfactual == 1).hi .- @rsubset(df_mean, :counterfactual == 1).avg, 
@rsubset(df_mean, :counterfactual == 1).avg .- @rsubset(df_mean, :counterfactual == 1).lo),
linecolor = "black",
label="Counterfactual owners",
linestyle = :dashdot)

plot!([0], 
seriestype = "vline", 
linestyle = :dash, 
linecolor = "red", 
label = "")

p1 = plot!()

df_mean = collapsed("tot_wealth")

plot(@rsubset(df_mean, :counterfactual == 0).event, 
@rsubset(df_mean,  :counterfactual == 0).avg,
ribbon = (@rsubset(df_mean, :counterfactual == 0).hi .- @rsubset(df_mean, :counterfactual == 0).avg, 
@rsubset(df_mean, :counterfactual == 0).avg .- @rsubset(df_mean, :counterfactual == 0).lo),
title = "B: Total Net Wealth",
xlabel = "Years from home purchase",
ylabel = "100,000 CHF",
linecolor = "black",
label="Owners",
xticks = -5:5:30,
ylimits = (0,10),
legend = :bottomright,
legend_font_pointsize = 11,
fontfamily = "Times Roman")

plot!(@rsubset(df_mean, :counterfactual == 1).event, 
@rsubset(df_mean, :counterfactual == 1).avg,
ribbon = (@rsubset(df_mean, :counterfactual == 1).hi .- @rsubset(df_mean, :counterfactual == 1).avg, 
@rsubset(df_mean, :counterfactual == 1).avg .- @rsubset(df_mean, :counterfactual == 1).lo),
linecolor = "black",
label="Counterfactual owners",
linestyle = :dashdot)

plot!([0], 
seriestype = "vline", 
linestyle = :dash, 
linecolor = "red", 
label = "")

p2 = plot!()

plot(p1, p2, layour = (1,1))
plot!(size=(1000,500), left_margin=7mm, 
    bottom_margin = 8mm, top_margin = 7mm) 
cd(figures)
savefig("event_study_wealth.pdf")

#############
# figure 5
plot(rresults["MainResults_Megaloop"]["baseline_diffst"].group, 
rresults["MainResults_Megaloop"]["baseline_diffst"].effect,
ribbon = (rresults["MainResults_Megaloop"]["baseline_diffst"].upper .- rresults["MainResults_Megaloop"]["baseline_diffst"].effect, 
rresults["MainResults_Megaloop"]["baseline_diffst"].effect .- rresults["MainResults_Megaloop"]["baseline_diffst"].lower),
title = "A: Difference in Liquid Wealth",
xlabel = "Years from home purchase",
ylabel = "100,000 CHF",
linecolor = "black",
fillcolor = :gray,
label="Owners",
xticks = -5:5:30,
ylimits = (-1.5,1),
legend = false )

plot!([0], 
seriestype = "vline", 
linestyle = :dash, 
linecolor = "red", 
label = "")

p1 = plot!()

plot(rresults["MainResults_Megaloop"]["baseline_diffwt"].group, 
rresults["MainResults_Megaloop"]["baseline_diffwt"].effect,
ribbon = (rresults["MainResults_Megaloop"]["baseline_diffwt"].upper .- rresults["MainResults_Megaloop"]["baseline_diffwt"].effect, 
rresults["MainResults_Megaloop"]["baseline_diffwt"].effect .- rresults["MainResults_Megaloop"]["baseline_diffwt"].lower),
title = "B: Difference in Total Net Wealth",
xlabel = "Years from home purchase",
ylabel = "100,000 CHF",
linecolor = "black",
fillcolor = :gray,
label="Owners",
xticks = -5:5:30,
ylimits = (-1.5,1),
#yticks = -1.5:0.5:0.5,
legend = false )

plot!([0], 
seriestype = "vline", 
linestyle = :dash, 
linecolor = "red", 
label = "")

p2 = plot!()

plot(p1, p2, layour = (1,1))
plot!(size=(1000,500), left_margin=7mm, 
    bottom_margin = 8mm, top_margin = 7mm) 
cd(figures)
savefig("event_study_wealth_cs.pdf")


#############
# figure 6

df_mean = collapsed("tot_consumption")

plot(@rsubset(df_mean, :counterfactual == 0).event, 
@rsubset(df_mean,  :counterfactual == 0).avg,
ribbon = (@rsubset(df_mean, :counterfactual == 0).hi .- @rsubset(df_mean, :counterfactual == 0).avg, 
@rsubset(df_mean, :counterfactual == 0).avg .- @rsubset(df_mean, :counterfactual == 0).lo),
title = "A: Total Consumption",
xlabel = "Years from home purchase",
ylabel = "100,000 CHF",
linecolor = "black",
label="Owners",
xticks = -5:5:30,
ylimits = (0,1.25),
legend = :bottomright,
legend_font_pointsize = 11)

plot!(@rsubset(df_mean, :counterfactual == 1).event, 
@rsubset(df_mean, :counterfactual == 1).avg,
ribbon = (@rsubset(df_mean, :counterfactual == 1).hi .- @rsubset(df_mean, :counterfactual == 1).avg, 
@rsubset(df_mean, :counterfactual == 1).avg .- @rsubset(df_mean, :counterfactual == 1).lo),
linecolor = "black",
label="Counterfactual owners",
linestyle = :dashdot)

plot!([0], 
seriestype = "vline", 
linestyle = :dash, 
linecolor = "red", 
label = "")

p1 = plot!()

plot(rresults["MainResults_Megaloop"]["baseline_diffct"].group, 
rresults["MainResults_Megaloop"]["baseline_diffct"].effect,
ribbon = (rresults["MainResults_Megaloop"]["baseline_diffct"].upper .- rresults["MainResults_Megaloop"]["baseline_diffct"].effect, 
rresults["MainResults_Megaloop"]["baseline_diffct"].effect .- rresults["MainResults_Megaloop"]["baseline_diffct"].lower),
title = "B: Difference in Total Consumption",
xlabel = "Years from home purchase",
ylabel = "100,000 CHF",
linecolor = "black",
fillcolor = "#80CDFD",
label="Owners",
xticks = -5:5:30,
ylimits = (-0.1,0.4),
legend = false) 

plot!([0], 
seriestype = "vline", 
linestyle = :dash, 
linecolor = "red", 
label = "")

p2 = plot!()

plot(p1, p2, layour = (1,1))
plot!(size=(1000,500), left_margin=7mm, 
    bottom_margin = 8mm, top_margin = 7mm) 
cd(figures)
savefig("event_study_consumption.pdf")


######################################################################
### Portfolio composition (section 5)

#############
# figure 7
df_mean = collapsed("AP")

plot(@rsubset(df_mean, :counterfactual == 0).event, 
@rsubset(df_mean,  :counterfactual == 0).avg,
ribbon = (@rsubset(df_mean, :counterfactual == 0).hi .- @rsubset(df_mean, :counterfactual == 0).avg, 
@rsubset(df_mean, :counterfactual == 0).avg .- @rsubset(df_mean, :counterfactual == 0).lo),
title = "A: Participation to Stock Market",
xlabel = "Years from home purchase",
ylabel = "Share",
linecolor = "black",
label="Owners",
xticks = -5:5:30,
ylimits = (0,1),
legend = :bottomright,
legend_font_pointsize = 11)


plot!(@rsubset(df_mean, :counterfactual == 1).event, 
@rsubset(df_mean, :counterfactual == 1).avg,
ribbon = (@rsubset(df_mean, :counterfactual == 1).hi .- @rsubset(df_mean, :counterfactual == 1).avg, 
@rsubset(df_mean, :counterfactual == 1).avg .- @rsubset(df_mean, :counterfactual == 1).lo),
linecolor = "black",
label="Counterfactual owners",
linestyle = :dashdot)

plot!([0], 
seriestype = "vline", 
linestyle = :dash, 
linecolor = "red", 
label = "")

p1 = plot!()


function collapsed_cond(var) # Collapase conditional on participation
    var_sym = Symbol(var)
    df_owners = @rsubset(df, :owner == 1, :AP == 1)
    df_owners.event .= df_owners.time .- df_owners.purchase
    df_mean = combine(groupby(df_owners, [:event, :counterfactual]), var_sym .=> mean .=> "avg", var_sym .=> std .=> :se, nrow => :count, renamecols = false)
    df_mean = @rsubset(df_mean, :event >= -5, :event <= 30)
    
    df_mean.hi = df_mean.avg .+ 1.96 .* df_mean.se ./ sqrt.(df_mean.count)
    df_mean.lo = df_mean.avg .- 1.96 .* df_mean.se ./ sqrt.(df_mean.count)

    return(df_mean)
end

df_mean = collapsed_cond("AT")

plot(@rsubset(df_mean, :counterfactual == 0).event, 
@rsubset(df_mean,  :counterfactual == 0).avg,
ribbon = (@rsubset(df_mean, :counterfactual == 0).hi .- @rsubset(df_mean, :counterfactual == 0).avg, 
@rsubset(df_mean, :counterfactual == 0).avg .- @rsubset(df_mean, :counterfactual == 0).lo),
title = "B: Conditional Stock Shares in Portfolio",
xlabel = "Years from home purchase",
ylabel = "Share",
linecolor = "black",
label="Owners",
xticks = -5:5:30,
ylimits = (0,1),
legend = :topright,
legend_font_pointsize = 11)

plot!(@rsubset(df_mean, :counterfactual == 1).event, 
@rsubset(df_mean, :counterfactual == 1).avg,
ribbon = (@rsubset(df_mean, :counterfactual == 1).hi .- @rsubset(df_mean, :counterfactual == 1).avg, 
@rsubset(df_mean, :counterfactual == 1).avg .- @rsubset(df_mean, :counterfactual == 1).lo),
linecolor = "black",
label="Counterfactual owners",
linestyle = :dashdot)

plot!([0], 
seriestype = "vline", 
linestyle = :dash, 
linecolor = "red", 
label = "")

p2 = plot!()

df.atst .= df.AT .* df.ST 
df_mean = collapsed("atst")

plot(@rsubset(df_mean, :counterfactual == 0).event, 
@rsubset(df_mean,  :counterfactual == 0).avg,
ribbon = (@rsubset(df_mean, :counterfactual == 0).hi .- @rsubset(df_mean, :counterfactual == 0).avg, 
@rsubset(df_mean, :counterfactual == 0).avg .- @rsubset(df_mean, :counterfactual == 0).lo),
title = "C: Unconditional Volume of Stocks",
xlabel = "Years from home purchase",
ylabel = "100,000 CHF",
linecolor = "black",
label="Owners",
xticks = -5:5:30,
ylimits = (0,2.5),
legend = :topright,
legend_font_pointsize = 11)

plot!(@rsubset(df_mean, :counterfactual == 1).event, 
@rsubset(df_mean, :counterfactual == 1).avg,
ribbon = (@rsubset(df_mean, :counterfactual == 1).hi .- @rsubset(df_mean, :counterfactual == 1).avg, 
@rsubset(df_mean, :counterfactual == 1).avg .- @rsubset(df_mean, :counterfactual == 1).lo),
linecolor = "black",
label="Counterfactual owners",
linestyle = :dashdot)

plot!([0], 
seriestype = "vline", 
linestyle = :dash, 
linecolor = "red", 
label = "")

p3 = plot!()

df_mean = collapsed_cond("atst")

plot(@rsubset(df_mean, :counterfactual == 0).event, 
@rsubset(df_mean,  :counterfactual == 0).avg,
ribbon = (@rsubset(df_mean, :counterfactual == 0).hi .- @rsubset(df_mean, :counterfactual == 0).avg, 
@rsubset(df_mean, :counterfactual == 0).avg .- @rsubset(df_mean, :counterfactual == 0).lo),
title = "D: Conditional Volume of Stocks",
xlabel = "Years from home purchase",
ylabel = "100,000 CHF",
linecolor = "black",
label="Owners",
xticks = -5:5:30,
ylimits = (0,2.5),
legend = :topright,
legend_font_pointsize = 11)

plot!(@rsubset(df_mean, :counterfactual == 1).event, 
@rsubset(df_mean, :counterfactual == 1).avg,
ribbon = (@rsubset(df_mean, :counterfactual == 1).hi .- @rsubset(df_mean, :counterfactual == 1).avg, 
@rsubset(df_mean, :counterfactual == 1).avg .- @rsubset(df_mean, :counterfactual == 1).lo),
linecolor = "black",
label="Counterfactual owners",
linestyle = :dashdot)

plot!([0], 
seriestype = "vline", 
linestyle = :dash, 
linecolor = "red", 
label = "")

p4 = plot!()

df.atst_rel .= df.atst ./ df.tot_wealth 
df_mean = collapsed_cond("atst_rel")

plot(@rsubset(df_mean, :counterfactual == 0).event, 
@rsubset(df_mean,  :counterfactual == 0).avg,
ribbon = (@rsubset(df_mean, :counterfactual == 0).hi .- @rsubset(df_mean, :counterfactual == 0).avg, 
@rsubset(df_mean, :counterfactual == 0).avg .- @rsubset(df_mean, :counterfactual == 0).lo),
title = "E: Cond. Stock Shares in Total Net Wealth",
xlabel = "Years from home purchase",
ylabel = "Share",
linecolor = "black",
label="Owners",
xticks = -5:5:30,
ylimits = (0,0.25),
legend = :topright,
legend_font_pointsize = 11)

plot!(@rsubset(df_mean, :counterfactual == 1).event, 
@rsubset(df_mean, :counterfactual == 1).avg,
ribbon = (@rsubset(df_mean, :counterfactual == 1).hi .- @rsubset(df_mean, :counterfactual == 1).avg, 
@rsubset(df_mean, :counterfactual == 1).avg .- @rsubset(df_mean, :counterfactual == 1).lo),
linecolor = "black",
label="Counterfactual owners",
linestyle = :dashdot)

plot!([0], 
seriestype = "vline", 
linestyle = :dash, 
linecolor = "red", 
label = "")

p5 = plot!()

plot(rresults["MainResults_Megaloop"]["cond_part_diffstat_rel"].group, 
rresults["MainResults_Megaloop"]["cond_part_diffstat_rel"].effect,
ribbon = (rresults["MainResults_Megaloop"]["cond_part_diffstat_rel"].upper .- rresults["MainResults_Megaloop"]["cond_part_diffstat_rel"].effect, 
rresults["MainResults_Megaloop"]["cond_part_diffstat_rel"].effect .- rresults["MainResults_Megaloop"]["cond_part_diffstat_rel"].lower),
title = "F: Panel E, Difference",
xlabel = "Years from home purchase",
ylabel = "Share",
linecolor = "black",
fillcolor = "gray",
label="Owners",
xticks = -5:5:30,
ylimits = (-0.05,0.075),
legend = false)

plot!([0], 
seriestype = "vline", 
linestyle = :dash, 
linecolor = "red", 
label = "")

p6 = plot!()

plot(p1,p2, p3, p4, p5, p6, layout = (3,2))
plot!(size=(1000,1500), left_margin=7mm, 
bottom_margin = 8mm, top_margin = 7mm)
cd(figures)
savefig("event_participation6.pdf")

#############
# figure 8 

df_owners = @rsubset(df, :owner == 1, :purchase > 1)
df_owners.event .= df_owners.time .- df_owners.purchase

df_owners.flag1 =  ifelse.((df_owners.AP .== 1) .&& (df_owners.event .== -1), 1, 0)
transform!(groupby(df_owners, [:id]), :flag1 => maximum => :flag1)

# Agents participating before
df_mean = @rsubset(df_owners, :flag1 == 1)
df_mean.atst .= df_mean.AT .* df_mean.ST
df_mean = combine(groupby(df_mean, [:event, :counterfactual]), :atst .=> mean .=> "avg", :atst .=> std .=> :se, nrow => :count, renamecols = false)
df_mean.hi = df_mean.avg .+ 1.96 .* df_mean.se ./ sqrt.(df_mean.count)
df_mean.lo = df_mean.avg .- 1.96 .* df_mean.se ./ sqrt.(df_mean.count)
df_mean = @rsubset(df_mean, :event >= -5, :event <= 30)

plot(@rsubset(df_mean, :counterfactual == 0).event, 
@rsubset(df_mean,  :counterfactual == 0).avg,
ribbon = (@rsubset(df_mean, :counterfactual == 0).hi .- @rsubset(df_mean, :counterfactual == 0).avg, 
@rsubset(df_mean, :counterfactual == 0).avg .- @rsubset(df_mean, :counterfactual == 0).lo),
title = "A: Volume of Stocks, Cond. on Early Part.",
xlabel = "Years from home purchase",
ylabel = "100,000 CHF",
linecolor = "black",
label="Owners",
xticks = -5:5:30,
ylimits = (0,2.5),
legend = :topright,
legend_font_pointsize = 11, )

plot!(@rsubset(df_mean, :counterfactual == 1).event, 
@rsubset(df_mean, :counterfactual == 1).avg,
ribbon = (@rsubset(df_mean, :counterfactual == 1).hi .- @rsubset(df_mean, :counterfactual == 1).avg, 
@rsubset(df_mean, :counterfactual == 1).avg .- @rsubset(df_mean, :counterfactual == 1).lo),
linecolor = "black",
label="Counterfactual owners",
linestyle = :dashdot)

plot!([0], 
seriestype = "vline", 
linestyle = :dash, 
linecolor = "red", 
label = "")

p1 = plot!()

plot(rresults["MainResults_Megaloop"]["cond_prior_part_diffstat"].group, 
rresults["MainResults_Megaloop"]["cond_prior_part_diffstat"].effect,
ribbon = (rresults["MainResults_Megaloop"]["cond_prior_part_diffstat"].upper .- rresults["MainResults_Megaloop"]["cond_prior_part_diffstat"].effect, 
rresults["MainResults_Megaloop"]["cond_prior_part_diffstat"].effect .- rresults["MainResults_Megaloop"]["cond_prior_part_diffstat"].lower),
title = "B: Panel A, Difference",
xlabel = "Years from home purchase",
ylabel = "100,000 CHF",
linecolor = "black",
fillcolor = "gray",
label="Owners",
xticks = -5:5:30,
ylimits = (-0.1,0.1),
#yticks = -0.3:0.05:0.3,
legend = false)

plot!([0], 
seriestype = "vline", 
linestyle = :dash, 
linecolor = "red", 
label = "")

p2 = plot!()

plot(p1, p2, layout = (1,2))
plot!(size=(1000,500), left_margin=7mm, 
    bottom_margin = 8mm, top_margin = 7mm) 
cd(figures)
savefig("event_participation_cs_pre.pdf")


######################################################################
### APPENDIX 

# figure A1
function collapsed(var)
    var_sym = Symbol(var)
    df_mean = combine(groupby(df, [:time, :owner, :counterfactual]), var_sym .=> mean .=> "avg", var_sym .=> std .=> :se, nrow => :count, renamecols = false)
    
    df_mean.hi = df_mean.avg .+ 1.96 .* df_mean.se ./ sqrt.(df_mean.count)
    df_mean.lo = df_mean.avg .- 1.96 .* df_mean.se ./ sqrt.(df_mean.count)

    return(df_mean)
end

### WELATH OVER LIFE CYCLE 
df_mean = collapsed("ST")
df_mean.age .= df_mean.time .+ 24

plot(@rsubset(df_mean, :owner == 1, :counterfactual == 0).age, 
@rsubset(df_mean, :owner == 1, :counterfactual == 0).avg,
ribbon = (@rsubset(df_mean, :owner == 1, :counterfactual == 0).hi .- @rsubset(df_mean, :owner == 1, :counterfactual == 0).avg, 
@rsubset(df_mean, :owner == 1, :counterfactual == 0).avg .- @rsubset(df_mean, :owner == 1, :counterfactual == 0).lo),
title = "A: Liquid Wealth",
xlabel = "Age",
ylabel = "100,000 CHF",
linecolor = "black",
label="Owners",
xticks = 25:10:120,
ylimits = (0,10),
legend = :topleft,
legend_font_pointsize = 11)

plot!(@rsubset(df_mean, :owner == 1, :counterfactual == 1).age, 
@rsubset(df_mean, :owner == 1, :counterfactual == 1).avg,
ribbon = (@rsubset(df_mean, :owner == 1, :counterfactual == 1).hi .- @rsubset(df_mean, :owner == 1, :counterfactual == 1).avg, 
@rsubset(df_mean, :owner == 1, :counterfactual == 1).avg .- @rsubset(df_mean, :owner == 1, :counterfactual == 1).lo),
linecolor = "black",
label="Counterfactual owners",
linestyle = :dashdot)

plot!(@rsubset(df_mean, :owner == 0, :counterfactual == 0).age, 
@rsubset(df_mean, :owner == 0, :counterfactual == 0).avg,
ribbon = (@rsubset(df_mean, :owner == 0, :counterfactual == 0).hi .- @rsubset(df_mean, :owner == 0, :counterfactual == 0).avg, 
@rsubset(df_mean, :owner == 0, :counterfactual == 0).avg .- @rsubset(df_mean, :owner == 0, :counterfactual == 0).lo),
linecolor = "black",
label="Tenants",
linestyle = :dot)

p1 = plot!()

df_mean = collapsed("tot_wealth")
df_mean.age .= df_mean.time .+ 24

plot(@rsubset(df_mean, :owner == 1, :counterfactual == 0).age, 
@rsubset(df_mean, :owner == 1, :counterfactual == 0).avg,
ribbon = (@rsubset(df_mean, :owner == 1, :counterfactual == 0).hi .- @rsubset(df_mean, :owner == 1, :counterfactual == 0).avg, 
@rsubset(df_mean, :owner == 1, :counterfactual == 0).avg .- @rsubset(df_mean, :owner == 1, :counterfactual == 0).lo),
title = "B: Total Net Wealth",
xlabel = "Age",
ylabel = "100,000 CHF",
linecolor = "black",
label="Owners",
xticks = 25:10:120,
ylimits = (0,10),
legend = :topleft,
legend_font_pointsize = 11)

plot!(@rsubset(df_mean, :owner == 1, :counterfactual == 1).age, 
@rsubset(df_mean, :owner == 1, :counterfactual == 1).avg,
ribbon = (@rsubset(df_mean, :owner == 1, :counterfactual == 1).hi .- @rsubset(df_mean, :owner == 1, :counterfactual == 1).avg, 
@rsubset(df_mean, :owner == 1, :counterfactual == 1).avg .- @rsubset(df_mean, :owner == 1, :counterfactual == 1).lo),
linecolor = "black",
label="Counterfactual owners",
linestyle = :dashdot)

plot!(@rsubset(df_mean, :owner == 0, :counterfactual == 0).age, 
@rsubset(df_mean, :owner == 0, :counterfactual == 0).avg,
ribbon = (@rsubset(df_mean, :owner == 1, :counterfactual == 0).hi .- @rsubset(df_mean, :owner == 1, :counterfactual == 0).avg, 
@rsubset(df_mean, :owner == 1, :counterfactual == 0).avg .- @rsubset(df_mean, :owner == 1, :counterfactual == 0).lo),
linecolor = "black",
label="Tenants",
linestyle = :dot)

p2 = plot!()

plot(p1, p2, layour = (1,1))
plot!(size=(1000,500), left_margin=7mm, 
    bottom_margin = 8mm, top_margin = 7mm) 
cd(figures)
savefig("lifecycle_wealth120.pdf")


#############
# figure A2 

function collapsed(var) # Collapse by event time
    var_sym = Symbol(var)
    df_owners = @rsubset(df, :owner == 1, :purchase > 1)
    df_owners.event .= df_owners.time .- df_owners.purchase
    df_mean = combine(groupby(df_owners, [:event, :counterfactual]), var_sym .=> mean .=> "avg", var_sym .=> std .=> :se, nrow => :count, renamecols = false)
    df_mean = @rsubset(df_mean, :event >= -5, :event <= 30)
    
    df_mean.hi = df_mean.avg .+ 1.96 .* df_mean.se ./ sqrt.(df_mean.count)
    df_mean.lo = df_mean.avg .- 1.96 .* df_mean.se ./ sqrt.(df_mean.count)

    return(df_mean)
end

df_mean = collapsed("CT")

plot(@rsubset(df_mean, :counterfactual == 0).event, 
@rsubset(df_mean,  :counterfactual == 0).avg,
ribbon = (@rsubset(df_mean, :counterfactual == 0).hi .- @rsubset(df_mean, :counterfactual == 0).avg, 
@rsubset(df_mean, :counterfactual == 0).avg .- @rsubset(df_mean, :counterfactual == 0).lo),
title = "A: Non-housing Consumption",
xlabel = "Years from home purchase",
ylabel = "100,000 CHF",
linecolor = "black",
label="Owners",
xticks = -5:5:30,
ylimits = (0,1),
legend = :topright,
legend_font_pointsize = 11)

plot!(@rsubset(df_mean, :counterfactual == 1).event, 
@rsubset(df_mean, :counterfactual == 1).avg,
ribbon = (@rsubset(df_mean, :counterfactual == 1).hi .- @rsubset(df_mean, :counterfactual == 1).avg, 
@rsubset(df_mean, :counterfactual == 1).avg .- @rsubset(df_mean, :counterfactual == 1).lo),
linecolor = "black",
label="Counterfactual owners",
linestyle = :dashdot)

plot!([0], 
seriestype = "vline", 
linestyle = :dash, 
linecolor = "red", 
label = "")

p1 = plot!()

df_mean = collapsed("h_costs")

plot(@rsubset(df_mean, :counterfactual == 0).event, 
@rsubset(df_mean,  :counterfactual == 0).avg,
ribbon = (@rsubset(df_mean, :counterfactual == 0).hi .- @rsubset(df_mean, :counterfactual == 0).avg, 
@rsubset(df_mean, :counterfactual == 0).avg .- @rsubset(df_mean, :counterfactual == 0).lo),
title = "B: Housing Costs",
xlabel = "Years from home purchase",
ylabel = "100,000 CHF",
linecolor = "black",
label="Owners",
xticks = -5:5:30,
ylimits = (0,1),
legend = :topright,
legend_font_pointsize = 11)

plot!(@rsubset(df_mean, :counterfactual == 1).event, 
@rsubset(df_mean, :counterfactual == 1).avg,
ribbon = (@rsubset(df_mean, :counterfactual == 1).hi .- @rsubset(df_mean, :counterfactual == 1).avg, 
@rsubset(df_mean, :counterfactual == 1).avg .- @rsubset(df_mean, :counterfactual == 1).lo),
linecolor = "black",
label="Counterfactual owners",
linestyle = :dashdot)

plot!([0], 
seriestype = "vline", 
linestyle = :dash, 
linecolor = "red", 
label = "")

p2 = plot!()

plot(p1, p2, layour = (1,1))
plot!(size=(1000,500), left_margin=7mm, 
    bottom_margin = 8mm, top_margin = 7mm) 
cd(figures)
savefig("event_study_consumption_types.pdf")

#############
# figure A3

plot(rresults["MainResults_Megaloop"]["cond_part_diffat"].group, 
rresults["MainResults_Megaloop"]["cond_part_diffat"].effect,
ribbon = (rresults["MainResults_Megaloop"]["cond_part_diffat"].upper .- rresults["MainResults_Megaloop"]["cond_part_diffat"].effect, 
rresults["MainResults_Megaloop"]["cond_part_diffat"].effect .- rresults["MainResults_Megaloop"]["cond_part_diffat"].lower),
title = "Difference in Conditional Stock Shares",
xlabel = "Years from home purchase",
ylabel = "Share",
linecolor = "black",
fillcolor = "gray",
label="Owners",
xticks = -5:5:30,
ylimits = (-0.1,0.15),
#yticks = -0.3:0.05:0.3,
legend = false)

plot!([0], 
seriestype = "vline", 
linestyle = :dash, 
linecolor = "red", 
label = "")

p1 = plot!()

plot(p1, layout = (1,2))
plot!(size=(500,500), left_margin=7mm, 
bottom_margin = 8mm, top_margin = 7mm)
cd(figures)
savefig("event_participation_cs.pdf")


#############
# figure A4

plot(rresults["MainResults_Megaloop"]["cond_prior_part_diffcumsum"].group, 
rresults["MainResults_Megaloop"]["cond_prior_part_diffcumsum"].effect,
ribbon = (rresults["MainResults_Megaloop"]["cond_prior_part_diffcumsum"].upper .- rresults["MainResults_Megaloop"]["cond_prior_part_diffcumsum"].effect, 
rresults["MainResults_Megaloop"]["cond_prior_part_diffcumsum"].effect .- rresults["MainResults_Megaloop"]["cond_prior_part_diffcumsum"].lower),
title = "Difference in Cumulative Stock Returns",
xlabel = "Years from home purchase",
ylabel = "100,000 CHF",
linecolor = "black",
fillcolor = "gray",
label="Owners",
xticks = -5:5:30,
ylimits = (-0.1,0.1),
legend = false)

plot!([0], 
seriestype = "vline", 
linestyle = :dash, 
linecolor = "red", 
label = "")

p1 = plot!()

plot(p1, layout = (2,2))
plot!(size=(500,500), left_margin=7mm, 
    bottom_margin = 8mm, top_margin = 7mm) 
cd(figures)
savefig("event_participation_cs_pre_cumsum.pdf")
