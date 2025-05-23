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
    using StatsBase
    using Interpolations
    using CSV
    using Alert
end

####################################################################################################################
### Setup

begin # Set directories 
    background_data = raw"~\background_data"
    results = raw"~\results"
end 

begin # Parameters and background data
    points = 10 # Number of points in grids
    T= 70+25 # Length of life
    N= 40 #length of working life
    β = 0.97 # Discount parameter
    γ = 5 # Risk aversion parameter
    δ = 0.2 # Down payment rate
    η = 0.22 # Housing consumption preference parameter
    inv_eta = 1/(1-η) # Shortcut for 1/(1-η)
    r_free = 0.018 # Risk free interest rate
    R = 1 + r_free # Gross risk free interest rate
    mpb = 0.88 # Marginal propensity to consume
    θ = (mpb*R/(1-mpb))^(γ)/(β*R) # Bequest intensity parameter
    k = 1 # Bequest curvature parameter
    z = 0.8  # Pension replacement rate
    I_part = 0.01 # Stock market participation fee (relative to current income)
    I_entry = 0.06 # Stock market entry fee (relative to current income)
    Random.seed!(42)
end 

begin # Load data for housing prices
    cd(background_data)
    df_re = DataFrame(XLSX.readtable("re_indeces_prices.xlsx", "data"))

    df_re[!,:year] = string.(df_re[:,:year])
    df_re[!,:year] = parse.(Int, df_re[:,:year])

    ψ_r = mean(df_re.alpha_t) # Relative rental price

    # You can use the prices from the data. However, the series is very noisy. 
    pt = zeros(T) # Empty vector of housing prices
    Random.seed!(42)
    random = rand(Truncated(Normal(0,80), -10000, +10000), T)
    for t = 1:T
        pt[t] = 2463.248 + t*99.99899  + random[t] # Coefficients from simple OLS regression (using only years after 1977)
    end
end

begin # Create vector for mortality prob
    cd(background_data)
    df_mortality = DataFrame(XLSX.readtable("mortality.xlsx", "data"))
    df_mortality[!,:Age] = string.(df_mortality[:,:Age])
    df_mortality[!,:Age] = parse.(Int, df_mortality[:,:Age])
    df_mortality = df_mortality[(df_mortality.Age .> 25),:]
    df_mortality[!,:pi] = string.(df_mortality[:,:pi])
    π = parse.(Float64, df_mortality[:,:pi])
end

begin # Create income profile
    cd(background_data)
    y = DataFrame(XLSX.readtable("income.xlsx", "all_values"))

    y = y[:, [:year, :age_25, :age_35, :age_45, :age_55, :age_65]]
    y = @rsubset(y, :year > 1977)  # Drop 1977 because we don't have inflation data on this year
    y1 = Matrix(y[:, [:age_25, :age_35, :age_45, :age_55, :age_65]])
    rename!(y, [:"year", :"25", :"35", :"45", :"55", :"65"])
    y = stack(y, [:"25", :"35", :"45", :"55", :"65"], :year)
    rename!(y, [:variable, :value] .=> [:age, :avg_income])
    y[!, :year_standard] = y[!, :year] .- 1977
    y[!, :age_standard] = parse.(Float64, y[!, :age]) .- 24

    # Create interpolation for yearly income growth
    age_s = unique!(y[!, :age_standard])
    year_s = unique!(y[!, :year_standard])

    interp_y = extrapolate(interpolate((year_s,age_s,), y1, Gridded(Linear())), Interpolations.Flat())  # Two-dimensional interpolation

    ypsilon = zeros(N)
    ypsilon[1] = 1

    for t in 2:N
        ypsilon[t] = 1 + (interp_y(t, t) - interp_y(t - 1, t - 1)) / interp_y(t - 1, t - 1) # yearly growth
    end

    # Unusual increase in the last years before retirment. Interpolate using only the first 35 years of life
    time_vect = range(1,N-5)
    interp_y_corrected = extrapolate(interpolate((time_vect,), ypsilon[1:N-5], Gridded(Linear())), Interpolations.Flat())  # Two-dimensional interpolation

    for t in 2:N
        ypsilon[t] = interp_y_corrected(t)
    end

    mu = interp_y(1, 1) # mean income distribution for first year
    sd = interp_y(1, 1) * 0.6 # sd income distribution for first year
    lognormal_st = sqrt(log(1 + (sd/mu)^2)) # transform the values for lognormal distribution
    lognormal_mu = log(mu) - lognormal_st^2 / 2
 
    # Create possible income profiles
    first_income=  rand(LogNormal(lognormal_mu,lognormal_st), points) # draw  1000 starting incomes from lognormal distribution
    rndy= rand(Normal(0, 0.1,), (points,N)) 

    Y = zeros(Float64,(points, T)) # create matrix of income profiles
    Y[:,1] = first_income .* 12
    Y[:,1] = sort( Y[:,1])

    # Income profiles for working age
        for i in range(1, points)
            for t in range(1,N)
                Y[i, t+1] = (Y[i, t])*ypsilon[t]
            end
        end

    # Income profiles for retirement age
        for i in range(1, points)
            for t in range(N+1,T)
                Y[i,t] = Y[i,N]*z
            end
        end
        
    # Sort values
    for t in range(1,T)
        Y[:,t] = sort( Y[:,t])
    end

end

begin # Create matrix for liquid wealth
    S = zeros(Float64,(points))
    S[1] = 0

    for i = 2:points
        S[i] = S[i-1]+(2500 + 8000*(i-2)^2)
        println(S[i])
    end
end

begin # Create matrix for net housing value
    ω = range(-0.1,0.8,points)
end 

begin # Create a vector with possible values of owned real esatate 
    h_bar = range(30,120, length = points)
end

begin #  Create sets of possible interest rates and income

    #(1 - p) * (-0.05) + p * u = 0.027
    # u^2 * p + 0.05^2 * (1-p) - 0.027^2 = 0.04^2
    
    # Stocks
    upval  =   73/225
    downval =  -0.05  
    prob =   81/337 
    meanv = downval*(1-prob) + upval*prob
    ((downval-meanv)^2*(1-prob) + (upval-meanv)^2*prob)^0.5
    rt_backw = [downval 1/(1-prob)
            upval 1/prob]
    
    # Real estate
    prob = 1089/1105 
    upval =  23/660 
    downval = -0.3
    meanv = downval*(1-prob) + upval*prob
    ((downval-meanv)^2*(1-prob) + (upval-meanv)^2*prob)^0.5
 
    rh_backw = [downval 1/(1-prob)
    upval 1/prob
    -1/100 2
    7/100 2  ]

    # Income shocks
    downval =  -0.3 # -0.05 
    upval =  593/13600  #193/3600 
    prob = 4624/4673 # 324/373
    meanv = downval*(1-prob) + upval*prob
    ((downval-meanv)^2*(1-prob) + (upval-meanv)^2*prob)^0.5

    y_shocks = [downval 1/(1-prob)
    upval 1/(prob)]

end

begin # Rescale vecotrs and matrices 
    S[:] .= S[:]/10000
    Y[:, :] .= Y[:, :]/10000
    pt[:] .= pt[:]/10000 
end

###########################################################################################################
### Define functions 
# FYI: eopa means "end of period assets"

function h_c(ct, RE, τ) # Derive housing consumption for renters/owners
    if RE == 0
        ht = ct * (η/(1-η)) / (pt[T-τ] * ψ_r)
    else 
        ht = RE
    end
    return(ht)
end

function q(pval) # Total consumption costs for renters
    q = ((η/(1 - η))/(pval* ψ_r))^η
    return(q)
end

function u_c(ct, hval, pval) # Utility function
    if hval == 0 
        u_val = (ct * q(pval))^(1 - γ)/(1 - γ) 
    else 
        u_val = (ct^(1 - η) * hval^η)^(1 - γ)/(1 - γ)
    end
    return(u_val)
end

function v_w(w, ω_val, h, pval) # Utility of bequest  
    if h != 0
        v_w_val = (w + h_bar[h]*ω_val*pval + k)^(1-γ)/(1-γ)
    else
        v_w_val = (w + k)^(1-γ)/(1-γ)
    end
    return(v_w_val)
end

function a_c(eopa, y, at, o, hval, τ) # First derivative of future utility + utility of bequest wrt of a 

    ψ_o = ψ_r 

    e1 = 0
    e2 = 0
    
    if T-τ < N
        for ys = 1:lastindex(y_shocks[:,1]) 
            for rt = 1:lastindex(rt_backw[:,1])
                for rh = 1:lastindex(rh_backw[1,:])
                    if hval == 0
                        pt1 = pt[T-τ] * (1 + rh_backw[rh+2, 1])
                        e1 +=  1/(rt_backw[rt, 2]*rh_backw[rh+2, 2]*y_shocks[ys, 2]) * (Y[y, T-τ] * (1 + y_shocks[ys, 1]) + eopa * (1 + r_free) + eopa * (rt_backw[rt, 1] - r_free) * at - mean(Y[:, T-τ])*I_part * at)^(-γ) * (q(pt1) / inv_eta)^(1-γ) * ((rt_backw[rt, 1] - r_free) * eopa - mean(Y[:, T-τ])*I_part)
                        e2 +=  1/(rt_backw[rt, 2]*rh_backw[rh+2, 2]*y_shocks[ys, 2]) * (eopa * (1 + r_free) + eopa * (rt_backw[rt, 1] - r_free) * at + k - mean(Y[:, T-τ])*I_part * at)^(-γ) * ((rt_backw[rt, 1] - r_free) * eopa - mean(Y[:, T-τ])*I_part)
                        if e1 == -Inf 
                            e1 = -10e1
                        end
                        if e2 == -Inf 
                            e2 = -10e1
                        end
                    else
                        pt1 = pt[T-τ]*(1 + rh_backw[rh, 1]) 
                        ω1 = (pt1/pt[T-τ] - (1-ω[o]))/(pt1/pt[T-τ])
                        Ct1_p = Y[y, T-τ] * (1 + y_shocks[ys, 1]) + eopa * (1 + r_free) + eopa * (rt_backw[rt, 1] - r_free) * at - hval * (1 - ω1) * pt[T-τ] *ψ_o  - mean(Y[:, T-τ])*I_part * at
                        if Ct1_p <= 0
                            e1 +=   -10e1
                        else 
                            e1 +=  1/(rt_backw[rt, 2]*rh_backw[rh, 2]*y_shocks[ys, 2]) * Ct1_p^((1-η)*(1-γ)-1) * hval^(η*(1-γ))  * (1-η) * ((rt_backw[rt, 1] - r_free) * eopa - mean(Y[:, T-τ])*I_part)
                        end
                        e2 +=  1/(rt_backw[rt, 2]*rh_backw[rh, 2]*y_shocks[ys, 2]) * (eopa * (1 + r_free) + eopa * (rt_backw[rt, 1] - r_free) * at + hval * ω1 * pt[T-τ] + k - mean(Y[:, T-τ])*I_part * at)^(-γ) * ((rt_backw[rt, 1] - r_free) * eopa - mean(Y[:, T-τ])*I_part)
                    end
                end
            end
        end
    elseif T-τ == N
        for rt = 1:lastindex(rt_backw[:,1])
            for rh = 1:lastindex(rh_backw[1,:])
                if hval == 0
                    pt1 = pt[T-τ] * (1 + rh_backw[rh+2, 1]) 
                    e1 +=  1/(rt_backw[rt, 2]*rh_backw[rh+2, 2]) * ((Y[y, T-τ] * z + eopa * (1 + r_free) + eopa * (rt_backw[rt, 1] - r_free) * at - mean(Y[:, T-τ])*I_part * at))^(-γ) * (q(pt1) / inv_eta)^(1-γ) * ((rt_backw[rt, 1] - r_free) * eopa - mean(Y[:, T-τ])*I_part)
                    e2 +=  1/(rt_backw[rt, 2]*rh_backw[rh+2, 2]) * (eopa * (1 + r_free) + eopa * (rt_backw[rt, 1] - r_free) * at + k - mean(Y[:, T-τ])*I_part * at)^(-γ) * ((rt_backw[rt, 1] - r_free) * eopa - mean(Y[:, T-τ])*I_part)
                    if e1 == -Inf 
                        e1 = -10e1
                    end
                    if e2 == -Inf 
                        e2 = -10e1
                    end
                else
                    pt1 = pt[T-τ]*(1 + rh_backw[rh, 1]) 
                    ω1 = (pt1/pt[T-τ] - (1-ω[o]))/(pt1/pt[T-τ])
                    Ct1_p = Y[y, T-τ] * z + eopa * (1 + r_free) + eopa * (rt_backw[rt, 1] - r_free) * at - hval * (1 - ω1) * pt[T-τ] *ψ_o  - mean(Y[:, T-τ])*I_part * at
                    if Ct1_p <= 0
                        e1 += -10e1
                    else 
                        e1 +=  1/(rt_backw[rt, 2]*rh_backw[rh, 2]) * Ct1_p^((1-η)*(1-γ)-1) * hval^(η*(1-γ))  * (1-η) * ((rt_backw[rt, 1] - r_free) * eopa - mean(Y[:, T-τ])*I_part)
                    end
                    e2 +=  1/(rt_backw[rt, 2]*rh_backw[rh, 2]) * (eopa * (1 + r_free) + eopa * (rt_backw[rt, 1] - r_free) * at + hval * ω1 * pt[T-τ] +  k - mean(Y[:, T-τ])*I_part * at)^(-γ) * ((rt_backw[rt, 1] - r_free) * eopa - mean(Y[:, T-τ])*I_part)
                end
            end
        end
    elseif T-τ > N && T-τ < T
        for rt = 1:lastindex(rt_backw[:,1])
            for rh = 1:lastindex(rh_backw[1,:])
                if hval == 0
                    pt1 = pt[T-τ] * (1 + rh_backw[rh+2, 1]) 
                    e1 +=  1/(rt_backw[rt, 2]*rh_backw[rh+2, 2]) * ((Y[y, T-τ] + eopa * (1 + r_free) + eopa * (rt_backw[rt, 1] - r_free) * at - mean(Y[:, T-τ])*I_part * at))^(-γ) * (q(pt1) / inv_eta)^(1-γ) * ((rt_backw[rt, 1] - r_free) * eopa - mean(Y[:, T-τ])*I_part)
                    e2 +=  1/(rt_backw[rt, 2]*rh_backw[rh+2, 2]) * (eopa * (1 + r_free) + eopa * (rt_backw[rt, 1] - r_free) * at + k - mean(Y[:, T-τ])*I_part * at)^(-γ) * ((rt_backw[rt, 1] - r_free) * eopa - mean(Y[:, T-τ])*I_part)
                    if e1 == -Inf 
                        e1 = -10e1
                    end
                    if e2 == -Inf 
                        e2 = -10e1
                    end
                else
                    pt1 = pt[T-τ]*(1 + rh_backw[rh, 1]) 
                    ω1 = (pt1/pt[T-τ] - (1-ω[o]))/(pt1/pt[T-τ])
                    Ct1_p = Y[y, T-τ] + eopa * (1 + r_free) + eopa * (rt_backw[rt, 1] - r_free) * at - hval * (1 - ω1) * pt[T-τ] *ψ_o  - mean(Y[:, T-τ])*I_part * at
                    if Ct1_p <= 0
                        e1 += -10e1
                    else 
                        e1 +=  1/(rt_backw[rt, 2]*rh_backw[rh, 2]) * Ct1_p^((1-η)*(1-γ)-1) * hval^(η*(1-γ))  * (1-η) * ((rt_backw[rt, 1] - r_free) * eopa - mean(Y[:, T-τ])*I_part)
                    end
                    e2 +=  1/(rt_backw[rt, 2]*rh_backw[rh, 2]) * (eopa * (1 + r_free) + eopa * (rt_backw[rt, 1] - r_free) * at + hval * ω1 * pt[T-τ] +  k - mean(Y[:, T-τ])*I_part * at)^(-γ) * ((rt_backw[rt, 1] - r_free) * eopa - mean(Y[:, T-τ])*I_part)
                end
            end
        end
    elseif T-τ == T
        for rt = 1:lastindex(rt_backw[:,1])
            for rh = 1:lastindex(rh_backw[1,:])
                if hval == 0
                    e2 +=  1/(rt_backw[rt, 2]*rh_backw[rh+2, 2]) * (eopa * (1 + r_free) + eopa * (rt_backw[rt, 1] - r_free) * at + k - mean(Y[:, T-τ])*I_part * at)^(-γ) * ((rt_backw[rt, 1] - r_free) * eopa - mean(Y[:, T-τ])*I_part)
                    if e2 == -Inf 
                        e2 = -10e1
                    end
                else
                    pt1 = pt[T-τ]*(1 + rh_backw[rh, 1]) 
                    ω1 = (pt1/pt[T-τ] - (1-ω[o]))/(pt1/pt[T-τ])
                    e2 +=  1/(rt_backw[rt, 2]*rh_backw[rh, 2]) * (eopa * (1 + r_free) + eopa * (rt_backw[rt, 1] - r_free) * at + hval * ω1 * pt[T-τ] +  k - mean(Y[:, T-τ])*I_part * at)^(-γ) * ((rt_backw[rt, 1] - r_free) * eopa - mean(Y[:, T-τ])*I_part)
                end
            end
        end
    end
    e = (1-π[T-τ]) * β * e1 + (π[T-τ]) * θ* β * e2
    return(e)
end

function atzero(eopa, y, o, hval, τ) # Find value of a which minimizes first derivative 
    try 
        at = fzero(x -> a_c(eopa, y, x, o, hval, τ), 0, Order0())

        if at > 0 && at < 1
            return(at)
        elseif at > 1 
            at = 1
            return(at)
        elseif at < 0 
            at = 0 
            return(at)
        end
        return(at)
    catch error 
        if isa(error, Roots.ConvergenceFailed)
            try 
                at = fzero( x -> a_c(eopa, y, x, o, hval, τ), 0, 1)
                return(at)
            catch error
                if isa(error, ArgumentError)
                    if a_c(eopa, y, 0, o, hval, τ) <= 0 &&  a_c(eopa, y, 1, o, hval, τ) <= 0
                        at = 0
                    elseif a_c(eopa, y, 0, o, hval, τ) > 0 &&  a_c(eopa, y, 1, o, hval, τ) > 0
                        at = 1
                    end
                end
                return(at)
            end       
        end
    end
end

function a_c_p(eopa, y, at, hval, τ) # First derivative of future utility + utility of bequest wrt of a, for purchasers
    e1 = 0
    e2 = 0
    ψ_o = ψ_r
    if T-τ < N
        for ys = 1:lastindex(y_shocks[:,1])
            for rt = 1:lastindex(rt_backw[:,1])
                for rh = 1:lastindex(rh_backw[1,:])
                    if hval == 0
                        pt1 = pt[T-τ] * (1 + rh_backw[rh+2, 1])
                        e1 += 1/(rt_backw[rt, 2]*rh_backw[rh+2, 2]*y_shocks[ys, 2])  * (Y[y, T-τ] * (1 + y_shocks[ys, 1]) + eopa * (1 + r_free) + eopa * (rt_backw[rt, 1] - r_free) * at - mean(Y[:, T-τ])*I_part * at)^(-γ) * (q(pt1) / inv_eta)^(1-γ) *((rt_backw[rt, 1] - r_free) * eopa - mean(Y[:, T-τ])*I_part)
                        e2 += 1/(rt_backw[rt, 2]*rh_backw[rh+2, 2]*y_shocks[ys, 2])  * (eopa * (1 + r_free) + eopa * (rt_backw[rt, 1] - r_free) * at + k - mean(Y[:, T-τ])*I_part * at)^(-γ) * ((rt_backw[rt, 1] - r_free) * eopa - mean(Y[:, T-τ])*I_part)
                        if e1 == -Inf 
                            e1 = -10e1
                        end
                        if e2 == -Inf 
                            e2 = -10e1
                        end
                    else
                        pt1 = pt[T-τ]*(1 + rh_backw[rh, 1]) 
                        ω1 =   (pt1/pt[T-τ] - (1-δ))/(pt1/pt[T-τ])
                        Ct1_p = Y[y, T-τ] * (1 + y_shocks[ys, 1]) + eopa * (1 + r_free) + eopa * (rt_backw[rt, 1] - r_free) * at - hval * (1 - ω1) * pt[T-τ] *ψ_o - mean(Y[:, T-τ])*I_part * at
                        if Ct1_p < 0 
                            e1 +=  -10e1
                        else
                            e1 += 1/(rt_backw[rt, 2]*rh_backw[rh, 2]*y_shocks[ys, 2])  * (Ct1_p)^((1-η)*(1-γ)-1) * hval^(η*(1-γ))  * (1-η) * ((rt_backw[rt, 1] - r_free) * eopa - mean(Y[:, T-τ])*I_part)
                        end
                        e2 += 1/(rt_backw[rt, 2]*rh_backw[rh, 2]*y_shocks[ys, 2])  * (eopa * (1 + r_free) + eopa * (rt_backw[rt, 1] - r_free) * at + hval * ω1 * pt[T-τ] + k - mean(Y[:, T-τ])*I_part * at)^(-γ) * ((rt_backw[rt, 1] - r_free) * eopa - mean(Y[:, T-τ])*I_part)
                    end
                end
            end
        end
    else
        for rt = 1:lastindex(rt_backw[:,1])
            for rh = 1:lastindex(rh_backw[1,:])
                if hval == 0
                    pt1 = pt[T-τ] * (1 + rh_backw[rh+2, 1])
                    e1 += 1/(rt_backw[rt, 2]*rh_backw[rh+2, 2])  * (Y[y, T-τ] * z + eopa * (1 + r_free) + eopa * (rt_backw[rt, 1] - r_free) * at - mean(Y[:, T-τ])*I_part * at)^(-γ) * (q(pt1) / inv_eta)^(1-γ) *((rt_backw[rt, 1] - r_free) * eopa - mean(Y[:, T-τ])*I_part)
                    e2 += 1/(rt_backw[rt, 2]*rh_backw[rh+2, 2])  * (eopa * (1 + r_free) + eopa * (rt_backw[rt, 1] - r_free) * at + k - mean(Y[:, T-τ])*I_part * at)^(-γ) * ((rt_backw[rt, 1] - r_free) * eopa - mean(Y[:, T-τ])*I_part)
                    if e1 == -Inf 
                        e1 = -10e1
                    end
                    if e2 == -Inf 
                        e2 = -10e1
                    end
                else
                    pt1 = pt[T-τ]*(1 + rh_backw[rh, 1]) 
                    ω1 =   (pt1/pt[T-τ] - (1-δ))/(pt1/pt[T-τ])
                    Ct1_p = Y[y, T-τ] * z + eopa * (1 + r_free) + eopa * (rt_backw[rt, 1] - r_free) * at - hval * (1 - ω1) * pt[T-τ] *ψ_o - mean(Y[:, T-τ])*I_part * at
                    if Ct1_p < 0 
                        e1 +=  -10e1
                    else
                        e1 += 1/(rt_backw[rt, 2]*rh_backw[rh, 2])  * (Ct1_p)^((1-η)*(1-γ)-1) * hval^(η*(1-γ))  * (1-η) * ((rt_backw[rt, 1] - r_free) * eopa - mean(Y[:, T-τ])*I_part)
                    end
                    e2 += 1/(rt_backw[rt, 2]*rh_backw[rh, 2])  * (eopa * (1 + r_free) + eopa * (rt_backw[rt, 1] - r_free) * at + hval * ω1 * pt[T-τ] + k - mean(Y[:, T-τ])*I_part * at)^(-γ) * ((rt_backw[rt, 1] - r_free) * eopa - mean(Y[:, T-τ])*I_part)
                end
            end
        end
    end
    e = (1-π[T-τ]) * β * e1 + (π[T-τ]) * θ* β * e2
    return(e)
end

function atzero_p(eopa, y, hval, τ) # Find value of a which minimizes first derivative 
    try 
        at = fzero(x -> a_c_p(eopa, y, x, hval, τ), 0, Order0())

        if at > 0 && at < 1
            return(at)
        elseif at > 1 
            at = 1
            return(at)
        elseif at < 0 
            at = 0 
            return(at)
        end
        return(at)
    catch error 
        if isa(error, Roots.ConvergenceFailed)
            try 
                at = fzero( x -> a_c_p(eopa, y, x, hval, τ), 0, 1)
                return(at)
            catch error
                if isa(error, ArgumentError)
                    if a_c_p(eopa, y, 0, hval, τ) <= 0 &&  a_c_p(eopa, y, 1, hval, τ) <= 0
                        at = 0
                    elseif a_c_p(eopa, y, 0, hval, τ) > 0 &&  a_c_p(eopa, y, 1, hval, τ) > 0
                        at = 1
                    end
                end
                return(at)
            end       
        end
    end
end

function second_upper_envelope_step(m_temp, c_temp, v_temp) # Second upper envelope refinement
    j1 = Int[]
    j2 = Int[]
    j3 = Int[]
    vupper1 = Int[]
    vupper2 = Int[]
    vupper3 = Int[]
    for i in 1:lastindex(m_temp)-1
        if m_temp[i] > m_temp[i+1]
            j1 = m_temp[1:i]
            for j in i+1:lastindex(m_temp)-1
                if m_temp[j] < m_temp[j+1] 
                    j2 = m_temp[i:j]
                    for k in j+1:lastindex(m_temp)-1 # Look for other kinks
                        if m_temp[k] > m_temp[k+1] # If there is another kink, don't take all vector till the end
                            j3 = m_temp[j:k]
                            vupper1 = extrapolate(interpolate((j1,),v_temp[1:i], Gridded(Linear())), Interpolations.Flat()) 
                            j2r = reverse(j2)
                            v2r = reverse(v_temp[i:j])
                            vupper2 = extrapolate(interpolate((j2r,), v2r, Gridded(Linear())), Interpolations.Flat())
                            vupper3 = extrapolate(interpolate((j3,), v_temp[j:k], Gridded(Linear())), Interpolations.Flat())
                            for jel in 1:lastindex(j2)
                                if v_temp[i+jel-1] < max(vupper1(j2[jel]), vupper2(j2[jel]), vupper3(j2[jel]))
                                    m_temp = deleteat!(m_temp, i+jel-1)
                                    v_temp = deleteat!(v_temp, i+jel-1)
                                    c_temp = deleteat!(c_temp, i+jel-1)
                                end
                            end
                            break
                        else
                            if k == lastindex(m_temp)-1
                                j3 = m_temp[j:lastindex(m_temp)]
                                j2r = reverse(j2)
                                v2r = reverse(v_temp[i:j])
                                vupper2 = extrapolate(interpolate((j2r,), v2r, Gridded(Linear())), Interpolations.Flat())
                                vupper3 = extrapolate(interpolate((j3,), v_temp[j:lastindex(m_temp)], Gridded(Linear())), Interpolations.Flat())
                                for jel in 1:lastindex(j2)
                                    if length(j1) > 1
                                        vupper1 = extrapolate(interpolate((j1,),v_temp[1:i], Gridded(Linear())), Interpolations.Flat()) 
                                        if v_temp[i+jel-1] < max(vupper1(j2[jel]), vupper2(j2[jel]), vupper3(j2[jel]))
                                            m_temp = deleteat!(m_temp, i+jel-1)
                                            v_temp = deleteat!(v_temp, i+jel-1)
                                            c_temp = deleteat!(c_temp, i+jel-1)
                                        end
                                    else 
                                        vupper1 =  v_temp[i]
                                        if v_temp[i+jel-1] < max( v_temp[i], vupper2(j2[jel]), vupper3(j2[jel]))
                                            m_temp = deleteat!(m_temp, i+jel-1)
                                            v_temp = deleteat!(v_temp, i+jel-1)
                                            c_temp = deleteat!(c_temp, i+jel-1)
                                        end
                                    end
                                end
                            end
                        end
                    end
                    break
                else
                    continue 
                end
            end
            break 
        else
            continue 
        end
    end
    return(m_temp, c_temp, v_temp)
end

function second_upper_envelope(m_temp, c_temp, v_temp) # Repeat SUE until vectors are clear
    length_original = length(m_temp)
    second_upper_envelope_step(m_temp,c_temp ,v_temp)
    length_new = length(m_temp)
    while length_original > length_new
        length_original = length(m_temp)
        second_upper_envelope_step(m_temp, c_temp, v_temp)     
        length_new = length(m_temp)
    end
    return(m_temp, c_temp, v_temp)
end

function dcegm_o(grid, y, ap, o, h, τ) # Owners

    ψ_o = ψ_r

    eopa_range = grid 

    c_temp = zeros(length(grid))
    m_temp = zeros(length(grid))
    v_temp = zeros(length(grid))
    rh = zeros(length(grid))
    EV = zeros(length(grid))

    c0 = zeros(length(grid))
    a0 = zeros(length(grid))
    v0 = zeros(length(grid))

    for (i,r) in enumerate(eopa_range) 
        if ap == 1
            at_r = atzero(r, y , o, h_bar[h], τ)
        else 
            at_r = 0 
        end

        #Initialize rh and EV value for ith eopa
        rh_i = 0 
        EV_i  = 0 
        # Iterate over each state of nature to find future levels of assets
        if T-τ < N
            for ys = 1:lastindex(y_shocks[:,1])
                for rt = 1:lastindex(rt_backw[:,1])
                    for rh = 1:lastindex(rh_backw[1,:])
                        Yt1 = Y[y, T-τ]*(1.0 + y_shocks[ys,1])
                        Lt1 = r * (1+r_free) * (1 - at_r) + r * (1 + rt_backw[rt, 1]) * at_r - mean(Y[:, T-τ])*I_part * at_r
                        pt1 = pt[T-τ]*(1 + rh_backw[rh, 1]) 
                        ω1 = (pt1/pt[T-τ] - (1-ω[o]))/(pt1/pt[T-τ])

                        if Yt1 + Lt1 - h_bar[h] * (1 - ω1) * pt1 *ψ_o  > 0  && Lt1 + h_bar[h] * ω1 * pt1 >= 0
                            cplus_p = C_E_o[T-τ+1, :, :, ap, :, h] # Get grid of future consumption for all st and wrh levels 
                            citp_p = extrapolate(interpolate((Y[:, T-τ+1], grid, ω[:],), cplus_p, Gridded(Linear())), Interpolations.Flat()) # Consumption function based on future utility levels (as you use the inverse marginal utility) 
                            Ct1_p = citp_p(Yt1, Lt1, ω1)

                            vplus_p = V_E_o[T-τ+1, :, :, ap, :, h]  # Get grid of future utility for all st and wrh levels 
                            vitp_p = extrapolate(interpolate((Y[:, T-τ+1] , grid, ω[:],), vplus_p, Gridded(Linear())), Interpolations.Flat())  # Two-dimensional interpolation
                            Vt1_p = vitp_p(Yt1, Lt1, ω1)

                            # Now you can calculate the RHS for this specific state of nature 
                            rhs_p = (1 - π[T-τ]) * β * ((1 + r_free) + at_r  * (rt_backw[rt, 1] - r_free)) * (1-η) * (Ct1_p)^((1-η)*(1-γ)-1) * h_bar[h]^(η*(1-γ))  +
                            π[T-τ]*θ* β * ((1 + r_free) + at_r  * (rt_backw[rt, 1] - r_free))  * (Lt1 + h_bar[h] * ω1 * pt1 + k) ^(-γ)
                            
                            # Aggregate RHS
                            rh_i += rhs_p/(rt_backw[rt, 2]*rh_backw[rh, 2]*y_shocks[ys, 2])
                            
                            # Aggregate EV
                            EV_i += ((1 - π[T-τ]) * β *  Vt1_p + π[T-τ] * θ * β * v_w(Lt1, ω1, h, pt1))/(rt_backw[rt, 2]*rh_backw[rh, 2]*y_shocks[ys, 2])
                            
                        else
                            rh_i += 0
                            EV_i += -10e1/(rt_backw[rt, 2]*rh_backw[rh, 2]*y_shocks[ys, 2]) 
                        end
                    end 
                end
            end
        elseif T-τ == N
            for rt = 1:lastindex(rt_backw[:,1])
                for rh = 1:lastindex(rh_backw[1,:])
                    Yt1 =  Y[y, T-τ]* z
                    Lt1 = r * (1+r_free) * (1 - at_r) + r * (1 + rt_backw[rt, 1]) * at_r - mean(Y[:, T-τ])*I_part * at_r
                    pt1 = pt[T-τ]*(1 + rh_backw[rh, 1]) 
                    ω1 = (pt1/pt[T-τ] - (1-ω[o]))/(pt1/pt[T-τ])

                    if Yt1 + Lt1 - h_bar[h] * (1 - ω1) * pt1 *ψ_o  > 0  && Lt1 + h_bar[h] * ω1 * pt1 >= 0
                        cplus_p = C_E_o[T-τ+1, :, :, ap, :, h] # Get grid of future consumption for all st and wrh levels 
                        citp_p = extrapolate(interpolate((Y[:, T-τ+1], grid, ω[:],), cplus_p, Gridded(Linear())), Interpolations.Flat()) # Consumption function based on future utility levels (as you use the inverse marginal utility) 
                        Ct1_p = citp_p(Yt1, Lt1, ω1)

                        vplus_p = V_E_o[T-τ+1, :, :, ap, :, h]  # Get grid of future utility for all st and wrh levels 
                        vitp_p = extrapolate(interpolate((Y[:, T-τ+1] , grid, ω[:],), vplus_p, Gridded(Linear())), Interpolations.Flat())  # Two-dimensional interpolation
                        Vt1_p = vitp_p(Yt1, Lt1, ω1)

                        # Now you can calculate the RHS for this specific state of nature 
                        rhs_p = (1 - π[T-τ]) * β * ((1 + r_free) + at_r  * (rt_backw[rt, 1] - r_free)) * (1-η) * (Ct1_p)^((1-η)*(1-γ)-1) * h_bar[h]^(η*(1-γ))  +
                            π[T-τ]*θ* β * ((1 + r_free) + at_r  * (rt_backw[rt, 1] - r_free))  * (Lt1 + h_bar[h] * ω1 * pt1 + k) ^(-γ)
                        
                            # Aggregate RHS
                        rh_i += rhs_p/(rt_backw[rt, 2]*rh_backw[rh, 2])

                        # Aggregate EV
                        EV_i += ((1 - π[T-τ]) * β *  Vt1_p + π[T-τ]*θ* β * v_w(Lt1, ω1, h, pt1))/(rt_backw[rt, 2]*rh_backw[rh, 2])
                    else
                        rh_i += 0
                        EV_i += -10e1/(rt_backw[rt, 2]*rh_backw[rh, 2]) 
                    end
                end
            end 
        elseif T-τ > N && T-τ < T
            for rt = 1:lastindex(rt_backw[:,1])
                for rh = 1:lastindex(rh_backw[1,:])
                    Yt1 = Y[y, T-τ+1]
                    Lt1 = r * (1+r_free) * (1 - at_r) + r * (1 + rt_backw[rt, 1]) * at_r - mean(Y[:, T-τ])*I_part * at_r
                    pt1 = pt[T-τ]*(1 + rh_backw[rh, 1]) 
                    ω1 = (pt1/pt[T-τ] - (1-ω[o]))/(pt1/pt[T-τ])

                    if Yt1 + Lt1 - h_bar[h] * (1 - ω1) * pt1 *ψ_o  > 0 && Lt1 + h_bar[h] * ω1 * pt1 >= 0

                        cplus_p = C_E_o[T-τ+1, :, :, ap, :, h] # Get grid of future consumption for all st and wrh levels 
                        citp_p = extrapolate(interpolate((Y[:, T-τ+1], grid, ω[:],), cplus_p, Gridded(Linear())), Interpolations.Flat()) # Consumption function based on future utility levels (as you use the inverse marginal utility) 
                        Ct1_p = citp_p(Yt1, Lt1, ω1)

                        vplus_p = V_E_o[T-τ+1, :, :, ap, :, h]  # Get grid of future utility for all st and wrh levels 
                        vitp_p = extrapolate(interpolate((Y[:, T-τ+1] , grid, ω[:],), vplus_p, Gridded(Linear())), Interpolations.Flat())  # Two-dimensional interpolation
                        Vt1_p = vitp_p(Yt1, Lt1, ω1)

                        # Now you can calculate the RHS for this specific state of nature 
                        rhs_p = (1 - π[T-τ]) * β * ((1 + r_free) + at_r  * (rt_backw[rt, 1] - r_free)) * (1-η) * (Ct1_p)^((1-η)*(1-γ)-1) * h_bar[h]^(η*(1-γ))  +
                            π[T-τ]*θ* β * ((1 + r_free) + at_r  * (rt_backw[rt, 1] - r_free))  * (Lt1 + h_bar[h] * ω1 * pt1 + k) ^(-γ)
                        
                            # Aggregate RHS
                        rh_i += rhs_p/(rt_backw[rt, 2]*rh_backw[rh, 2])

                        # Aggregate EV
                        EV_i += ((1 - π[T-τ]) * β *  Vt1_p + π[T-τ]*θ* β * v_w(Lt1, ω1, h, pt1))/(rt_backw[rt, 2]*rh_backw[rh, 2])
                    else     
                        rh_i += 0
                        EV_i += -10e1/(rt_backw[rt, 2]*rh_backw[rh, 2]) 
                    end
                end
            end 
        elseif T-τ == T
            for rt = 1:lastindex(rt_backw[:,1])
                for rh = 1:lastindex(rh_backw[1,:])
                    Lt1 = r * (1+r_free) * (1 - at_r) + r * (1 + rt_backw[rt, 1]) * at_r - mean(Y[:, T-τ])*I_part * at_r 
                    pt1 = pt[T-τ]*(1 + rh_backw[rh, 1]) 
                    ω1 = (pt1/pt[T-τ] - (1-ω[o]))/(pt1/pt[T-τ])

                    if Lt1 + h_bar[h] *  ω1 * pt1 >= 0 
                        # Now you can calculate the RHS for this specific state of nature 
                        rhs_p = θ * β * ((1 + r_free) + at_r  * (rt_backw[rt, 1] - r_free))  * (Lt1 + h_bar[h] * ω1 * pt1 + k) ^(-γ)

                        # Aggregate RHS
                        rh_i += rhs_p/(rt_backw[rt, 2]*rh_backw[rh, 2])

                        # Aggregate EV
                        EV_i += (θ* β * v_w(Lt1, ω1, h, pt1))/(rt_backw[rt, 2]*rh_backw[rh, 2])
                    else
                        rh_i += 0
                        EV_i += -10e1/(rt_backw[rt, 2]*rh_backw[rh, 2]) 
                    end
                end
            end 
        end

        rh[i] = rh_i
        EV[i] = EV_i
        expon = 1/((1-η) * (1-γ) - 1)
        if rh[i] > 0 
            c_temp[i]  = (h_bar[h] ^ (-η * (1-γ)) * (rh[i] * inv_eta)) ^ (expon) # Endogenous level of consumption
            m_temp[i] = r + c_temp[i] + h_bar[h] * (1 - ω[o]) * pt[T-τ] *ψ_o  + mean(Y[:, T-τ])*I_part * at_r # Endogenous level of resources
            v_temp[i] =  u_c(c_temp[i], h_bar[h], pt[T-τ]) + EV[i] 
        else 
            c_temp[i] = 1/10000
            m_temp[i] = 1/10000
            v_temp[i] =  -10e1
        end
        if m_temp[i] < 0 || c_temp[i] < 0  || v_temp[i] < -10e1
            c_temp[i] = 1/10000
            m_temp[i] = 1/10000
            v_temp[i] = -10e1
        end
    end

    unique!(c_temp)
    unique!(m_temp)
    unique!(v_temp)

    # There should be no kinks, but to be sure run a second upper envelope refinement (interpolations are sometimes a bit sloppy)
    sue = second_upper_envelope(m_temp, c_temp, v_temp)
    m_temp = sue[1]
    c_temp = sue[2]
    v_temp = sue[3]

    # Extrapolate consumption based on common grid for liquid assets
    c_extrapol = extrapolate(
        interpolate(([0, m_temp...],), [0, c_temp...], Gridded(Linear())), Interpolations.Flat()) # Recover consumption levels from endogenous grid
    v_extrapol = extrapolate(
        interpolate(([m_temp...],), [v_temp...], Gridded(Linear())), Interpolations.Flat()) 
    
    for (i,r) in enumerate(Y[y, T-τ] .+  grid) 
        c0[i] = c_extrapol(r)
        if ap == 1
            a0[i] = atzero(r - c0[i]  -h_bar[h] * (1 - ω[o]) * pt[T-τ] *ψ_o , y, o, h_bar[h], τ)  
        else
            a0[i]  = 0 
    end
        v0[i] = v_extrapol(r)
        if v0[i] == NaN || v0[i] == -Inf
            v0[i] = -10e1
        end
    end
    return(c0, a0, v0)
end

function dcegm_rr(grid, y, ap, τ) # Retired renters

    eopa_range = grid 

    c_temp = zeros(length(grid))
    m_temp = zeros(length(grid))
    v_temp = zeros(length(grid))
    rh = zeros(length(grid))
    EV = zeros(length(grid))

    c0 = zeros(length(grid))
    h0 = zeros(length(grid))
    a0 = zeros(length(grid))
    v0 = zeros(length(grid))

    for (i,r) in enumerate(eopa_range) 
        
        if ap == 1 
            at_r = atzero(r, y , 0,  0, τ)
        else 
            at_r = 0 
        end
        
        # Initialize rh and EV value for ith eopa 
        rh_i = 0 
        EV_i = 0

        # Iterate over each state of nature to find future levels of assets
        if T-τ != N && T-τ < T 
            for rt = 1:lastindex(rt_backw[:,1])
                Yt1 =  Y[y, T-τ]
                Lt1 = r * (1+r_free) * (1 - at_r) + r * (1 + rt_backw[rt, 1]) * at_r - mean(Y[:, T-τ])*I_part * at_r

                if Lt1 < 0 
                    # Aggregate RHS
                    rh_i += 0
                    # Aggregate EV
                    EV_i += -10e1/(rt_backw[rt, 2])
                else
                    for rh = 1:lastindex(rh_backw[1,:])
                        pt1 = pt[T-τ] * (1 + rh_backw[rh+2, 1])
                        cplus_r = C_E_r[T-τ+1, : , :, ap] 
                        citp_r = extrapolate(interpolate((Y[:, T-τ+1] , grid), cplus_r, Gridded(Linear())), Interpolations.Flat()) # Consumption function based on future utility levels (as you use the inverse marginal utility) 
                        Ct1_r = citp_r(Yt1, Lt1)

                        vplus_r = V_E_r[T-τ+1, :, :, ap]
                        vitp_r = extrapolate(interpolate((Y[:, T-τ+1], grid), vplus_r, Gridded(Linear())), Interpolations.Flat())
                        Vt1_r = vitp_r(Yt1, Lt1)

                        # Now you can calculate the RHS for this specific state of nature 
                        rhs_r = (1 - π[T-τ]) * β * ((1 + r_free) + at_r  * (rt_backw[rt, 1] - r_free))  * q(pt1) * (Ct1_r * q(pt1)) ^(-γ) + 
                            π[T-τ]*θ* β * ((1 + r_free) + at_r  * (rt_backw[rt, 1] - r_free))  * (Lt1 + k) ^(-γ)
                        
                        # Aggregate RHS
                        rh_i += rhs_r/(rt_backw[rt, 2]*rh_backw[rh+2, 2])
                    
                        # Aggregate EV
                        EV_i += ((1 - π[T-τ]) * β * Vt1_r + π[T-τ] * θ * β * v_w(Lt1, 0, 0, pt1))/(rt_backw[rt, 2]*rh_backw[rh+2, 2])
                    end
                end
            end 
        elseif T-τ ==  N         
            for rt = 1:lastindex(rt_backw[:,1])
                Yt1 =  Y[y, T-τ]* z
                Lt1 = r * (1+r_free) * (1 - at_r) + r * (1 + rt_backw[rt, 1]) * at_r - mean(Y[:, T-τ])*I_part * at_r
                
                if Lt1 < 0 
                    # Aggregate RHS
                    rh_i += 0
                    # Aggregate EV
                    EV_i += -10e1/(rt_backw[rt, 2]) 
                else
                    for rh = 1:lastindex(rh_backw[1,:])
                        pt1 = pt[T-τ] * (1 + rh_backw[rh+2, 1])
                        
                        cplus_r = C_E_r[T-τ+1, : , :, ap] 
                        citp_r = extrapolate(interpolate((Y[:, T-τ+1], grid), cplus_r, Gridded(Linear())), Interpolations.Flat()) # Consumption function based on future utility levels (as you use the inverse marginal utility) 
                        Ct1_r = citp_r(Yt1, Lt1)

                        vplus_r = V_E_r[T-τ+1, :, :, ap]
                        vitp_r = extrapolate(interpolate((Y[:, T-τ+1], grid), vplus_r, Gridded(Linear())), Interpolations.Flat())
                        Vt1_r = vitp_r(Yt1, Lt1)

                        # Now you can calculate the RHS for this specific state of nature 
                        rhs_r = (1 - π[T-τ]) * β * ((1 + r_free) + at_r  * (rt_backw[rt, 1] - r_free))  * q(pt1) * (Ct1_r * q(pt1)) ^(-γ) + 
                            π[T-τ]*θ* β * ((1 + r_free) + at_r  * (rt_backw[rt, 1] - r_free))   * (Lt1 + k) ^(-γ)
                        
                        # Aggregate RHS
                        rh_i += rhs_r/(rt_backw[rt, 2]*rh_backw[rh+2, 2])

                        # Aggregate EV
                        EV_i += (((1 - π[T-τ]) * β *  Vt1_r) + π[T-τ]*θ* β * v_w(Lt1, 0, 0, pt1))/(rt_backw[rt, 2]*rh_backw[rh+2, 2])
                    end
                end
            end
        elseif T-τ ==  T
            for rt = 1:lastindex(rt_backw[:,1])
                Lt1 = r * (1+r_free) * (1 - at_r) + r * (1 + rt_backw[rt, 1]) * at_r - mean(Y[:, T-τ])*I_part * at_r
                if Lt1 < 0
                    # Aggregate RHS
                    rh_i += 0
                    # Aggregate EV
                    EV_i += -10e1/(rt_backw[rt, 2]) 
                else                        
                    # Now you can calculate the RHS for this specific state of nature 
                    rhs_r = π[T-τ]*θ* β * ((1 + r_free) + at_r  * (rt_backw[rt, 1] - r_free))   * (Lt1 + k) ^(-γ)
                    
                    # Aggregate RHS
                    rh_i += rhs_r/(rt_backw[rt, 2])

                    # Aggregate EV
                    EV_i += (π[T-τ]*θ* β * v_w(Lt1, 0, 0, 0))/(rt_backw[rt, 2])
                end
            end
        end 

        rh[i] = rh_i
        EV[i] = EV_i

        if  rh[i] > 0 
            c_temp[i]  = ((1/q(pt[T-τ])^(1-γ)) * rh[i]) ^ (-1/γ) # Apply inverse euler 
            m_temp[i] = r + c_temp[i]  + h_c(c_temp[i], 0, τ) * ψ_r *  pt[T-τ] + mean(Y[:, T-τ])*I_part * at_r # Endogenous level of resources
            v_temp[i] =  u_c(c_temp[i], 0, pt[T-τ]) + EV[i]
        else 
            c_temp[i]  = 1/10000
            m_temp[i] = 1/10000
            v_temp[i] = -10e1 
        end
        if m_temp[i] < 0 || c_temp[i] < 0  || v_temp[i] < -10e1
            c_temp[i] = 1/10000
            m_temp[i] = 1/10000
            v_temp[i] = -10e1
        end
    end

    # Extrapolate consumption based on common grid for liquid assets
    c_extrapol = extrapolate(
       interpolate(([0, m_temp...],), [0,c_temp...], Gridded(Linear())), Interpolations.Flat()) # Recover consumption levels from endogenous grid\
    v_extrapol = extrapolate(
        interpolate(([m_temp...],), [v_temp...], Gridded(Linear())), Interpolations.Flat()) 
        
    for (i,r) in enumerate(Y[y, T-τ] .+ grid)
        c0[i] = c_extrapol(r)
        h0[i] = h_c(c0[i], 0, τ) 
        if ap == 1    
            a0[i] = atzero(r - c0[i]  - h0[i] * pt[T-τ] * ψ_r, y, 0,  0, τ)  
        else 
            a0[i] =  0
        end 
        v0[i] = v_extrapol(r)
    end
    return(c0, h0, a0, v0)
end

function dcegm_r(grid, y, ap, τ) # Working renters

    eopa_range = grid

    c_temp = zeros(length(grid))
    v_temp = zeros(length(grid))
    m_temp = zeros(length(grid))
    rh = zeros(length(grid))
    EV = zeros(length(grid))
    c0 = zeros(length(grid))
    h0 = zeros(length(grid))
    a0 = zeros(length(grid))
    v0 = zeros(length(grid))

    for (i,r) in enumerate(eopa_range) # For each EOPA we  need the RHS for both cases (renter, purchaser)
        
        if ap == 1 
            at_r = atzero(r, y , 0,  0, τ)
        else 
            at_r = 0
        end

        # Initialize rh and EV value for ith eopa
        rh_i = 0 
        EV_i  = 0 

        # Iterate over each state of nature to find future levels of assets
        for ys = 1:lastindex(y_shocks[:, 1])
            for rt = 1:lastindex(rt_backw[:,1])
                Yt1 = Y[y, T-τ]*(1.0 + y_shocks[ys, 1])
                Lt1 = r * (1+r_free) * (1 - at_r) + r * (1 + rt_backw[rt, 1]) * at_r - mean(Y[:, T-τ])*I_part * at_r # Future resources 
                
                if Lt1 < 0
                    EV_i += -10e1/(rt_backw[rt, 2]*y_shocks[ys, 2]) 
                    rh_i += 0

                else
                    for rh = 1:lastindex(rh_backw[1,:])
                        pt1 = pt[T-τ] * (1 + rh_backw[rh+2, 1])

                        # If we stay renters:
                        cplus_r = C_E_r[T-τ+1, :, :, ap]
                        citp_r = extrapolate(interpolate((Y[:, T-τ+1], grid), cplus_r, Gridded(Linear())), Interpolations.Flat())
                        Ct1_r = citp_r(Yt1, Lt1)

                        vplus_r = V_E_r[T-τ+1, :, :, ap]
                        vitp_r = extrapolate(interpolate((Y[:, T-τ+1], grid), vplus_r, Gridded(Linear())), Interpolations.Flat())
                        Vt1_r = vitp_r(Yt1, Lt1)
                    
                        rhs_r = (1 - π[T-τ]) * β * ((1 + r_free) + at_r  * (rt_backw[rt, 1] - r_free))  * q(pt1) * (Ct1_r * q(pt1)) ^(-γ) + 
                                π[T-τ]*θ* β * ((1 + r_free) + at_r  * (rt_backw[rt, 1] - r_free)) * (Lt1 + k) ^(-γ)
                        
                        # If we purchase
                        cplus_p = C_E_p[T-τ+1, :, :, ap] # Get grid of future consumption for all st and wrh levels 
                        vplus_p = V_E_p[T-τ+1, :, :, ap]
                        hplus_p = H_E_p[T-τ+1, :, :, ap]
                        citp_p = extrapolate(interpolate((Y[:, T-τ+1], grid), cplus_p, Gridded(Linear())), Interpolations.Flat())
                        vitp_p = extrapolate(interpolate((Y[:, T-τ+1] , grid), vplus_p, Gridded(Linear())), Interpolations.Flat())
                        hitp_p = extrapolate(interpolate((Y[:, T-τ+1] , grid), hplus_p, Gridded(Constant())), Interpolations.Flat())
                        Ht1_p = hitp_p(Yt1, Lt1)
                        Ct1_p = citp_p(Yt1, Lt1)
                        if Yt1 + Lt1 - Ct1_p - Ht1_p*pt1*((ψ_r*(1-δ)+δ)) < 0 || Ht1_p < 30
                            Vt1_p = -10e1
                        else
                            Vt1_p = vitp_p(Yt1, Lt1)
                        end
                        if Vt1_p <= -10e1 || Ht1_p < 30
                            rhs_p = 0  
                        else 
                            rhs_p = (1 - π[T-τ]) * β * ((1 + r_free) + at_r  * (rt_backw[rt, 1] - r_free)) * (1-η) * (Ct1_p)^((1-η)*(1-γ)-1) * Ht1_p^(η*(1-γ))  +
                            π[T-τ]*θ* β * ((1 + r_free) + at_r  * (rt_backw[rt, 1] - r_free)) * (Lt1 +  k) ^(-γ)
                        end
                        
                        EV_i += ((1 - π[T-τ]) * β * maximum(last, (Vt1_p, Vt1_r)) + π[T-τ] * θ * β * v_w(Lt1, 0, 0, pt1))/(rt_backw[rt, 2]*rh_backw[rh+2, 2]*y_shocks[ys, 2])

                        # Aggregate RHS
                        rh_i += maximum(last, (rhs_p, rhs_r))/(rt_backw[rt, 2]*rh_backw[rh+2, 2]*y_shocks[ys, 2])
                    end
                end
            end
        end

        rh[i] = rh_i
        EV[i] = EV_i
        
        if  rh[i] > 0 
            c_temp[i] = ((1/q(pt[T-τ])^(1-γ)) * rh[i]) ^ (-1/γ) # Apply inverse euler 
            m_temp[i] = r + c_temp[i] + h_c(c_temp[i], 0, τ) * ψ_r * pt[T-τ] + mean(Y[:, T-τ])*I_part * at_r # Endogenous level of resources
            v_temp[i] = u_c(c_temp[i], 0, pt[T-τ]) + EV[i]
        else
            c_temp[i]  = 1/10000
            m_temp[i] = 1/10000
            v_temp[i] = -10e1 
        end
        if m_temp[i] < 0 || c_temp[i] < 0  || v_temp[i] < -10e1
            c_temp[i] = 1/10000
            m_temp[i] = 1/10000
            v_temp[i] = -10e1
        end
    end

    # Eliminate kinks: Second upper envelope refinement
    sue = second_upper_envelope(m_temp, c_temp, v_temp)
    m_temp = sue[1]
    c_temp = sue[2]
    v_temp = sue[3]

    # Extrapolate consumption based on common grid for liquid assets
    c_extrapol = extrapolate(
       interpolate(([0, m_temp...],), [0, c_temp...], Gridded(Linear())), Interpolations.Flat()) # Recover consumption levels from endogenous grid
    
    v_extrapol = extrapolate(
        interpolate(([m_temp...],), [v_temp...], Gridded(Linear())), Interpolations.Flat()) 

    for (i,r) in enumerate(Y[y, T-τ]  .+ grid)
        c0[i] = c_extrapol(r)
        h0[i] = h_c(c0[i], 0, τ)
        if ap == 1    
            a0[i] = atzero(r - c0[i]  - h0[i] * pt[T-τ] * ψ_r, y, 0,  0, τ)  
        else 
            a0[i] = 0 
        end 
        v0[i] = v_extrapol(r) 
    end
    return(c0, h0, a0, v0)
end

function dcegm_p(grid, y, ap, τ) # Purchasers

    ψ_o = ψ_r

    eopa_range = grid
    
    c_temp = zeros(length(grid))
    h_temp = zeros(length(grid))
    m_temp = zeros(length(grid))
    v_temp = zeros(length(grid))

    rht = zeros(length(grid))

    c0 = zeros(length(grid))
    a0 = zeros(length(grid))
    v0 = zeros(length(grid))
    h0 = zeros(length(grid))

    for (i,r) in enumerate(eopa_range)
        c_vec = zeros(length(h_bar))
        m_vec = zeros(length(h_bar))
        v_vec = zeros(length(h_bar))
        rhth = zeros(length(h_bar))
        Threads.@threads for h = 1:lastindex(h_bar) # Iterate over discrete values of real estate
            r_net = r - h_bar[h] * (δ + (1-δ)*ψ_o) * pt[T-τ]
            if r_net > 0 
                if ap == 1 
                    at_r = atzero_p(r_net, y, h_bar[h], τ)
                else 
                    at_r = 0 
                end

                #initialize rh and EV value for ith eopa
                rh_i = 0 
                EV_i  = 0 
                
                # Iterate over each state of nature to find future levels of assets
                if T-τ < N
                    for ys = 1:lastindex(y_shocks[:,1]) 
                        for rt = 1:lastindex(rt_backw[:,1])
                            for rh = 1:lastindex(rh_backw[1,:])
                                Yt1 = Y[y, T-τ]*(1.0 + y_shocks[ys,1])
                                Lt1 = r_net * (1+r_free) * (1 - at_r) + r_net * (1 + rt_backw[rt, 1]) * at_r - mean(Y[:, T-τ])*I_part * at_r
                                pt1 = pt[T-τ]*(1 + rh_backw[rh, 1]) 
                                ω1 = (pt1/pt[T-τ] - (1-δ))/(pt1/pt[T-τ]) #(1 + rh_backw[rh, 1])*δ # Stochastic net value 

                                if Yt1 + Lt1 - h_bar[h] * (1 - ω1) * pt1 *ψ_o > 0 && Lt1 + h_bar[h] * ω1 * pt1 >= 0
                                    cplus_p = C_E_o[T-τ+1, :, :, ap, :, h] # Get grid of future consumption for all st and wrh levels 
                                    citp_p = extrapolate(interpolate((Y[:, T-τ+1], grid, ω[:],), cplus_p, Gridded(Linear())), Interpolations.Flat()) # Consumption function based on future utility levels (as you use the inverse marginal utility) 
                                    Ct1_p = citp_p(Yt1, Lt1, ω1)

                                    vplus_p = V_E_o[T-τ+1, :, :, ap, :, h]  # Get grid of future utility for all st and wrh levels 
                                    vitp_p = extrapolate(interpolate((Y[:, T-τ+1] , grid, ω[:],), vplus_p, Gridded(Linear())), Interpolations.Flat())  # Two-dimensional interpolation
                                    Vt1_p = vitp_p(Yt1, Lt1, ω1)

                                    # Now you can calculate the RHS for this specific state of nature 
                                    rhs_p = (1 - π[T-τ]) * β * ((1 + r_free) + at_r  * (rt_backw[rt, 1] - r_free)) * (1-η) * (Ct1_p)^((1-η)*(1-γ)-1) * h_bar[h]^(η*(1-γ))  +
                                    π[T-τ]*θ* β * ((1 + r_free) + at_r * (rt_backw[rt, 1] - r_free)) * (Lt1 + h_bar[h] * ω1 * pt[T-τ]  + k) ^(-γ)
                                    
                                    # Aggregate RHS
                                    rh_i += rhs_p/(rt_backw[rt, 2]*rh_backw[rh, 2]*y_shocks[ys,2])

                                    # Aggregate EV
                                    EV_i += ((1 - π[T-τ]) * β *  Vt1_p + π[T-τ] * θ * β * v_w(Lt1, ω1, h, pt1))/(rt_backw[rt, 2]*rh_backw[rh, 2]*y_shocks[ys, 2])
                                else 
                                    rh_i += 0
                                    EV_i += -10e1/(rt_backw[rt, 2]*rh_backw[rh, 2]*y_shocks[ys, 2]) 
                                end
                            end
                        end
                    end
                elseif T-τ == N
                    for rt = 1:lastindex(rt_backw[:,1])
                        for rh = 1:lastindex(rh_backw[1,:])
                            Yt1 = Y[y, T-τ]* z
                            Lt1 = r_net * (1+r_free) * (1 - at_r) + r_net * (1 + rt_backw[rt, 1]) * at_r - mean(Y[:, T-τ])*I_part * at_r
                            pt1 = pt[T-τ]*(1 + rh_backw[rh, 1]) 
                            ω1 =   (pt1/pt[T-τ] - (1-δ))/(pt1/pt[T-τ]) #(1 + rh_backw[rh, 1])*δ # Stochastic net value 
                            
                            if Yt1 + Lt1 - h_bar[h] * (1-δ) * pt[T-τ] *ψ_o > 0 && Lt1 + h_bar[h] * ω1 * pt1 >= 0
                                cplus_p = C_E_o[T-τ+1, :, :, ap, :, h] # Get grid of future consumption for all st and wrh levels 
                                citp_p = extrapolate(interpolate((Y[:, T-τ+1], grid, ω[:],), cplus_p, Gridded(Linear())), Interpolations.Flat()) # Consumption function based on future utility levels (as you use the inverse marginal utility) 
                                Ct1_p = citp_p(Yt1, Lt1, ω1)

                                vplus_p = V_E_o[T-τ+1, :, :, ap, :, h]  # Get grid of future utility for all st and wrh levels 
                                vitp_p = extrapolate(interpolate((Y[:, T-τ+1] , grid, ω[:],), vplus_p, Gridded(Linear())), Interpolations.Flat())  # Two-dimensional interpolation
                                Vt1_p = vitp_p(Yt1, Lt1, ω1)

                                # Now you can calculate the RHS for this specific state of nature 
                                rhs_p = (1 - π[T-τ]) * β * ((1 + r_free) + at_r  * (rt_backw[rt, 1] - r_free)) * (1-η) * (Ct1_p)^((1-η)*(1-γ)-1) * h_bar[h]^(η*(1-γ))  +
                                π[T-τ]*θ* β * ((1 + r_free) + at_r * (rt_backw[rt, 1] - r_free)) * (Lt1 + h_bar[h] * ω1 * pt[T-τ]  + k) ^(-γ)
                                
                                # Aggregate RHS
                                rh_i += rhs_p/(rt_backw[rt, 2]*rh_backw[rh, 2])

                                # Aggregate EV
                                EV_i += ((1 - π[T-τ]) * β *  Vt1_p + π[T-τ] * θ * β * v_w(Lt1, ω1, h, pt1))/(rt_backw[rt, 2]*rh_backw[rh, 2])
                            else 
                                rh_i += 0
                                EV_i += -10e1/(rt_backw[rt, 2]*rh_backw[rh, 2]) 
                            end
                        end
                    end
                end
                if rh_i <= 0 
                    rh_i = 0 
                    EV_i = -10e1
                end
                expon = 1/((1-η) * (1-γ) - 1)
                c_vec[h]  = (h_bar[h] ^ (-η * (1-γ)) * (rh_i * inv_eta)) ^ (expon)  
                m_vec[h] = r_net + c_vec[h] + h_bar[h] * (ψ_o*(1-δ)+δ) * pt[T-τ]  + mean(Y[:, T-τ])*I_part * at_r # Endogenous level of resources
                v_vec[h] = u_c(c_vec[h], h_bar[h], pt[T-τ]) + EV_i
                rhth[h] = rh_i
            else
                v_vec[h] = -10e1
            end
        end
    
        # Find highest utility among possible levels of real estate
        mi = findmax(v_vec)[2]
        c_temp[i]  = c_vec[mi]
        rht[i] = rhth[mi]
        m_temp[i] =  m_vec[mi]
        h_temp[i] =  h_bar[mi]
        v_temp[i] =  v_vec[mi]

    end

    # Extrapolate consumption based on common grid for liquid assets
    c_extrapol = extrapolate(
       interpolate(([0, m_temp...],), [0, c_temp...], Gridded(Linear())), Interpolations.Flat()) # Recover consumption levels from endogenous grid
    h_extrapol = extrapolate(
        interpolate(([0, m_temp...],), [0, h_temp...], Gridded(Constant())), Interpolations.Flat()) 
    v_extrapol = extrapolate(
        interpolate(([m_temp...],), [v_temp...], Gridded(Linear())), Interpolations.Flat()) 
    
    for (i,r) in enumerate(Y[y, T-τ] .+ grid)
        c0[i] = c_extrapol(r)
        h0[i] = h_extrapol(r)
        if ap == 1 
            a0[i] = atzero_p(r - c0[i] - pt[T-τ]*h0[i]*(ψ_o*(1-δ)+δ) , y, h0[i], τ)  
        else
            a0[i] = 0  
        end
        
        v0[i] = v_extrapol(r) 

       if c0[i] > r
            c0[i] = r
            h0[i] = 0
        end
   end
    return(c0, h0, a0, v0)
end

function dcegm_cr(grid, y, ap, τ) # Counterfactual renters

    eopa_range = grid 

    c_temp = zeros(length(grid))
    m_temp = zeros(length(grid))
    v_temp = zeros(length(grid))
    rh = zeros(length(grid))
    EV = zeros(length(grid))
    
    c0 = zeros(length(grid))
    h0 = zeros(length(grid))
    a0 = zeros(length(grid))
    v0 = zeros(length(grid))
    
    for (i,r) in enumerate(eopa_range) 
        
        if ap == 1 
            at_r = atzero(r, y , 0,  0, τ)
        else 
            at_r = 0 
        end
        
        # Initialize rh and EV value for ith eopa 
        rh_i = 0 
        EV_i = 0
    
        # Iterate over each state of nature to find future levels of assets
        if T-τ <  N 
            for ys = 1:lastindex(y_shocks[:,1])
                for rt = 1:lastindex(rt_backw[:,1])
                    Yt1 = Y[y, T-τ]*(1.0 + y_shocks[ys, 1])
                    Lt1 = r * (1+r_free) * (1 - at_r) + r * (1 + rt_backw[rt, 1]) * at_r - mean(Y[:, T-τ])*I_part * at_r
    
                    if Lt1 < 0 
                        # Aggregate RHS
                        rh_i += 0
                        # Aggregate EV
                        EV_i += -10e1/(rt_backw[rt, 2])
                    else
                        for rh = 1:lastindex(rh_backw[1,:])
                            pt1 = pt[T-τ] * (1 + rh_backw[rh+2, 1])
                            cplus_r = C_E_cr[T-τ+1, : , :, ap] 
                            citp_r = extrapolate(interpolate((Y[:, T-τ+1] , grid), cplus_r, Gridded(Linear())), Interpolations.Flat()) # Consumption function based on future utility levels (as you use the inverse marginal utility) 
                            Ct1_r = citp_r(Yt1, Lt1)
    
                            vplus_r = V_E_cr[T-τ+1, :, :, ap]
                            vitp_r = extrapolate(interpolate((Y[:, T-τ+1], grid), vplus_r, Gridded(Linear())), Interpolations.Flat())
                            Vt1_r = vitp_r(Yt1, Lt1)
    
                            # Now you can calculate the RHS for this specific state of nature 
                            rhs_r = (1 - π[T-τ]) * β * ((1 + r_free) + at_r  * (rt_backw[rt, 1] - r_free))  * q(pt1) * (Ct1_r * q(pt1)) ^(-γ) + 
                                π[T-τ]*θ* β * ((1 + r_free) + at_r  * (rt_backw[rt, 1] - r_free))  * (Lt1 + k) ^(-γ)
                            
                            # Aggregate RHS
                            rh_i += rhs_r/(rt_backw[rt, 2]*rh_backw[rh+2, 2]*y_shocks[ys, 2])
                        
                            # Aggregate EV
                            EV_i += ((1 - π[T-τ]) * β * Vt1_r + π[T-τ] * θ * β * v_w(Lt1, 0, 0, pt1))/(rt_backw[rt, 2]*rh_backw[rh+2, 2]*y_shocks[ys, 2])
                        end
                    end
                end 
            end
        elseif T-τ != N && T-τ < T 
            for rt = 1:lastindex(rt_backw[:,1])
                Yt1 =  Y[y, T-τ]
                Lt1 = r * (1+r_free) * (1 - at_r) + r * (1 + rt_backw[rt, 1]) * at_r - mean(Y[:, T-τ])*I_part * at_r
    
                if Lt1 < 0 
                    # Aggregate RHS
                    rh_i += 0
                    # Aggregate EV
                    EV_i += -10e1/(rt_backw[rt, 2])
                else
                    for rh = 1:lastindex(rh_backw[1,:])
                        pt1 = pt[T-τ] * (1 + rh_backw[rh+2, 1])
                        cplus_r = C_E_cr[T-τ+1, : , :, ap] 
                        citp_r = extrapolate(interpolate((Y[:, T-τ+1] , grid), cplus_r, Gridded(Linear())), Interpolations.Flat()) # Consumption function based on future utility levels (as you use the inverse marginal utility) 
                        Ct1_r = citp_r(Yt1, Lt1)
    
                        vplus_r = V_E_cr[T-τ+1, :, :, ap]
                        vitp_r = extrapolate(interpolate((Y[:, T-τ+1], grid), vplus_r, Gridded(Linear())), Interpolations.Flat())
                        Vt1_r = vitp_r(Yt1, Lt1)
    
                        # Now you can calculate the RHS for this specific state of nature 
                        rhs_r = (1 - π[T-τ]) * β * ((1 + r_free) + at_r  * (rt_backw[rt, 1] - r_free))  * q(pt1) * (Ct1_r * q(pt1)) ^(-γ) + 
                            π[T-τ]*θ* β * ((1 + r_free) + at_r  * (rt_backw[rt, 1] - r_free))  * (Lt1 + k) ^(-γ)
                        
                        # Aggregate RHS
                        rh_i += rhs_r/(rt_backw[rt, 2]*rh_backw[rh+2, 2])
                    
                        # Aggregate EV
                        EV_i += ((1 - π[T-τ]) * β * Vt1_r + π[T-τ] * θ * β * v_w(Lt1, 0, 0, pt1))/(rt_backw[rt, 2]*rh_backw[rh+2, 2])
                    end
                end
            end 
        elseif T-τ ==  N         
            for rt = 1:lastindex(rt_backw[:,1])
                Yt1 =  Y[y, T-τ]* z
                Lt1 = r * (1+r_free) * (1 - at_r) + r * (1 + rt_backw[rt, 1]) * at_r - mean(Y[:, T-τ])*I_part * at_r
                
                if Lt1 < 0 
                    # Aggregate RHS
                    rh_i += 0
                    # Aggregate EV
                    EV_i += -10e1/(rt_backw[rt, 2]) 
                else
                    for rh = 1:lastindex(rh_backw[1,:])
                        pt1 = pt[T-τ] * (1 + rh_backw[rh+2, 1])
                        
                        cplus_r = C_E_cr[T-τ+1, : , :, ap] 
                        citp_r = extrapolate(interpolate((Y[:, T-τ+1], grid), cplus_r, Gridded(Linear())), Interpolations.Flat()) # Consumption function based on future utility levels (as you use the inverse marginal utility) 
                        Ct1_r = citp_r(Yt1, Lt1)
    
                        vplus_r = V_E_cr[T-τ+1, :, :, ap]
                        vitp_r = extrapolate(interpolate((Y[:, T-τ+1], grid), vplus_r, Gridded(Linear())), Interpolations.Flat())
                        Vt1_r = vitp_r(Yt1, Lt1)
    
                        # Now you can calculate the RHS for this specific state of nature 
                        rhs_r = (1 - π[T-τ]) * β * ((1 + r_free) + at_r  * (rt_backw[rt, 1] - r_free))  * q(pt1) * (Ct1_r * q(pt1)) ^(-γ) + 
                            π[T-τ]*θ* β * ((1 + r_free) + at_r  * (rt_backw[rt, 1] - r_free))   * (Lt1 + k) ^(-γ)
                        
                        # Aggregate RHS
                        rh_i += rhs_r/(rt_backw[rt, 2]*rh_backw[rh+2, 2])
    
                        # Aggregate EV
                        EV_i += (((1 - π[T-τ]) * β *  Vt1_r) + π[T-τ]*θ* β * v_w(Lt1, 0, 0, pt1))/(rt_backw[rt, 2]*rh_backw[rh+2, 2])
                    end
                end
            end
        elseif T-τ ==  T
            for rt = 1:lastindex(rt_backw[:,1])
                Lt1 = r * (1+r_free) * (1 - at_r) + r * (1 + rt_backw[rt, 1]) * at_r - mean(Y[:, T-τ])*I_part * at_r
                if Lt1 < 0
                    # Aggregate RHS
                    rh_i += 0
                    # Aggregate EV
                    EV_i += -10e1/(rt_backw[rt, 2]) 
                else                        
                    # Now you can calculate the RHS for this specific state of nature 
                    rhs_r = π[T-τ]*θ* β * ((1 + r_free) + at_r  * (rt_backw[rt, 1] - r_free))   * (Lt1 + k) ^(-γ)
                    
                    # Aggregate RHS
                    rh_i += rhs_r/(rt_backw[rt, 2])
    
                    # Aggregate EV
                    EV_i += (π[T-τ]*θ* β * v_w(Lt1, 0, 0, 0))/(rt_backw[rt, 2])
                end
            end
        end 
    
        rh[i] = rh_i
        EV[i] = EV_i
    
        if  rh[i] > 0 
            c_temp[i]  = ((1/q(pt[T-τ])^(1-γ)) * rh[i]) ^ (-1/γ) # Apply inverse euler 
            m_temp[i] = r + c_temp[i]  + h_c(c_temp[i], 0, τ) * ψ_r *  pt[T-τ] + mean(Y[:, T-τ])*I_part * at_r # Endogenous level of resources
            v_temp[i] =  u_c(c_temp[i], 0, pt[T-τ]) + EV[i]
        else 
            c_temp[i]  = 1/10000
            m_temp[i] = 1/10000
            v_temp[i] = -10e1 
        end
        if m_temp[i] < 0 || c_temp[i] < 0  || v_temp[i] < -10e1
            c_temp[i] = 1/10000
            m_temp[i] = 1/10000
            v_temp[i] = -10e1
        end
    end

    # Extrapolate consumption based on common grid for liquid assets
    c_extrapol = extrapolate(
       interpolate(([0, m_temp...],), [0,c_temp...], Gridded(Linear())), Interpolations.Flat()) # Recover consumption levels from endogenous grid\
    v_extrapol = extrapolate(
        interpolate(([m_temp...],), [v_temp...], Gridded(Linear())), Interpolations.Flat()) 
        
    for (i,r) in enumerate(Y[y, T-τ] .+ grid)
        c0[i] = c_extrapol(r)
        h0[i] = h_c(c0[i], 0, τ) 
        if ap == 1    
            a0[i] = atzero(r - c0[i]  - h0[i] * pt[T-τ] * ψ_r, y, 0,  0, τ)  
        else 
            a0[i] =  0
        end 
        v0[i] = v_extrapol(r)
    end
    return(c0, h0, a0, v0)
end

####################################################################################################################
### Define empty grids 

begin # Grids for renters
    C_E_r = zeros(Float64,(T, points, points, 2))
    H_E_r = zeros(Float64,(T, points, points, 2))
    A_E_r = zeros(Float64,(T, points, points, 2))
    V_E_r = zeros(Float64,(T, points, points, 2))
    D_E_p = zeros(Float64,(T, points, points, 2))
    print("done")
end

begin # Grids for owners
    C_E_o = zeros(Float64,(T, points, points, 2, points, length(h_bar)))
    A_E_o = zeros(Float64,(T, points, points, 2, points, length(h_bar)))
    V_E_o = zeros(Float64,(T, points, points, 2, points, length(h_bar)))
    print("done")
end

begin # Grids for purchasers
    C_E_p= zeros(Float64,(T, points, points, 2))
    H_E_p = zeros(Float64,(T, points, points, 2))
    A_E_p = zeros(Float64,(T, points, points, 2))
    V_E_p = zeros(Float64,(T, points, points, 2))
    print("done")
end

begin # Grids for counterfactual renters
    C_E_cr = zeros(Float64,(T, points, points, 2))
    H_E_cr = zeros(Float64,(T, points, points, 2))
    A_E_cr = zeros(Float64,(T, points, points, 2))
    V_E_cr = zeros(Float64,(T, points, points, 2))
    print("done")
end

####################################################################################################################
### Backward induction - derive policy functions

@time begin #Owners
    for t =0:T-1
        println(t)
        for y in 1:points
            for ap in 1:2
                for o in 1:lastindex(ω)
                    Threads.@threads  for h in 1:lastindex(h_bar)
                        #println(y, " , ", ap, " , ",  o, " , ",  h, " , ", t)
                        result = dcegm_o(S[:], y, ap, o, h, t)
                        C_E_o[T-t, y, :, ap, o, h] = result[1] 

                        A_E_o[T-t, y, :, ap, o, h] = result[2] 

                        V_E_o[T-t, y, :, ap, o, h] = result[3]   
                    end
                end
            end
        end
    end
end

@time begin  # Older Renters
    for t = 0:T-N
        for y in 1:points
            for ap in 1:2
                result = dcegm_rr(S[:], y, ap, t)
                C_E_r[T-t, y, :, ap] = result[1]  

                H_E_r[T-t, y, :, ap] = result[2] 

                A_E_r[T-t, y, :, ap] = result[3] 

                V_E_r[T-t, y, :, ap] = result[4]      
            end
        end
    end
end

@time begin  # Purchasers
    for t = T-N:T-1
        for y in 1:points
            for ap in 1:2
                result = dcegm_p(S[:], y, ap, t)
                C_E_p[T-t, y, :, ap] = result[1]  

                H_E_p[T-t, y, :, ap] = result[2] 

                A_E_p[T-t, y, :, ap] = result[3] 

                V_E_p[T-t, y, :, ap] = result[4]      
            end
        end
    end
end

@time begin  # Working renters
    for t = T-N+1:T-1
        for y in 1:points
            for ap in 1:2
                result = dcegm_r(S[:], y, ap, t)
                C_E_r[T-t, y, :, ap] = result[1]  

                H_E_r[T-t, y, :, ap] = result[2] 

                A_E_r[T-t, y, :, ap] = result[3] 

                V_E_r[T-t, y, :, ap] = result[4]      
            end
        end
    end
end

@time begin  # Counterfactual renters
    for t = 0:T-1
        for y in 1:points
            for ap in 1:2
                result = dcegm_cr(S[:], y, ap, t)
                C_E_cr[T-t, y, :, ap] = result[1]  

                H_E_cr[T-t, y, :, ap] = result[2] 

                A_E_cr[T-t, y, :, ap] = result[3] 

                V_E_cr[T-t, y, :, ap] = result[4]      
            end
        end
    end
end

begin # Save data
    cd(results)
    save("C_E_o.jld", "data", C_E_o)
    save("A_E_o.jld", "data", A_E_o)
    save("V_E_o.jld", "data", V_E_o)
    save("C_E_r.jld", "data", C_E_r)
    save("A_E_r.jld", "data", A_E_r)
    save("V_E_r.jld", "data", V_E_r)
    save("H_E_r.jld", "data", H_E_r)
    save("C_E_p.jld", "data", C_E_p)
    save("A_E_p.jld", "data", A_E_p)
    save("V_E_p.jld", "data", V_E_p)
    save("H_E_p.jld", "data", H_E_p)
    save("C_E_cr.jld", "data", C_E_cr)
    save("A_E_cr.jld", "data", A_E_cr)
    save("V_E_cr.jld", "data", V_E_cr)
    save("H_E_cr.jld", "data", H_E_cr)

    println("saved!")
end 

###########################################################################################################
###  Forward induction - generate life cycle profiles

begin # Load data for housing prices
    vct = 1:T

    ptmat = vcat(vct', pt')'
    
    ptmat = DataFrame(ptmat, :auto)
    rename!(ptmat, "x1" => "time", "x2" => "pt")
    ptmat.time = Int.(ptmat.time)
end

function prob_d_f(VR, VP) # Purchase decision
    if VR>=VP 
        p0 = 0
    else
        p0 = 1      
    end
    return(p0)
end

function closest_value_s(t, val) # Refinement functions for defaulting agents (Get a closer value to the available ST values)
    dxfirst = val - S[1]
    dxindex = 1
    for (i,r) in enumerate(S[:])
        dx = val - r 
        if dx > 0 && dx < dxfirst
            dxindex = i
        end
    end
    return(S[dxindex])
end 

function closest_value_y(t, val) # Refinement functions for defaulting agents (Get a closer value to the available ST values)
    dxfirst = val - Y[1,t]
    dxindex = 1
    for (i,r) in enumerate(Y[:,t])
        dx = val - r 
        if dx > 0 && dx < dxfirst
            dxindex = i
        end
    end
    return(Y[dxindex,t])
end 

function df_creation(agent_n, seed_n, save) # Function for generating agents
    A = agent_n
    Random.seed!(seed_n)
    ψ_o = ψ_r
    RT = rand(Normal(0.06, 0.16,), (T))
    ####################################################################################################################
    ### Create matrices
    YT = zeros(Float64,(A, T)) # create matrix of income profiles
    CT = zeros(Float64,(A, T)) # create matrix of housing profiles
    HT = zeros(Float64,(A, T)) # create matrix of housing profiles
    ST = zeros(Float64,(A, T)) # create matrix of liquid assets profiles
    LT = zeros(Float64,(A, T)) # create matrix of cash at hand profiles
    AT = zeros(Float64,(A, T)) # create matrix of risky shares profiles
    AP = zeros(Float64,(A, T)) # create matrix of participation
    VT = zeros(Float64,(A, T)) # create matrix of utility profiles
    PR = zeros(Float64,(A, T)) # create matrix of purchasing probability
    OT = zeros(Float64,(A, T)) # create matrix of mortgages
    DT = zeros(Float64,(A, T)) # create matrix for ownership
    DP = zeros(Float64,(A, T)) # create matrix for purchase
    RG = zeros(Float64,(A, T)) # create matrix for stock returns
    PP = zeros(Float64,(A, T)) # create matrix for price at time of purchase
    DF = zeros(Float64,(A, T)) # create matrix for default
    ####################################################################################################################
    # Life-Cycle simulation
    # Workers
    nr = 0 
    for t in 1:T
        ## Interpolations
        #### NON PARTICIPATING
        # Renters
        cr_extp_t_np = extrapolate(interpolate((Y[:, t],S[:],), C_E_r[t, :, :, 2], Gridded(Linear())), Interpolations.Flat())  # Two-dimensional interpolation
        hr_extp_t_np = extrapolate(interpolate((Y[:, t],S[:],), H_E_r[t, :, :, 2], Gridded(Linear())), Interpolations.Flat())  # Two-dimensional interpolation
        ar_extp_t_np = extrapolate(interpolate((Y[:, t],S[:],), A_E_r[t, :, :, 2], Gridded(Linear())), Interpolations.Flat())  # Two-dimensional interpolation
        vr_extp_t_np = extrapolate(interpolate((Y[:, t],S[:],), V_E_r[t, :, :, 2], Gridded(Linear())), Interpolations.Flat())  # Two-dimensional interpolation

        # purchasers
        cp_extp_t_np = extrapolate(interpolate((Y[:, t],S[:],), C_E_p[t, :, :, 2], Gridded(Linear())), Interpolations.Flat())  # Two-dimensional interpolation
        hp_extp_t_np = extrapolate(interpolate((Y[:, t],S[:],), H_E_p[t, :, :, 2], Gridded(Constant())), Interpolations.Flat())  # Two-dimensional interpolation
        vp_extp_t_np = extrapolate(interpolate((Y[:, t],S[:],), V_E_p[t, :, :, 2], Gridded(Linear())), Interpolations.Flat())  # Two-dimensional interpolation
        ap_extp_t_np = extrapolate(interpolate((Y[:, t],S[:],), A_E_p[t, :, :, 2], Gridded(Linear())), Interpolations.Flat())  # Two-dimensional interpolation

        # Owners
        co_extp_t_np = extrapolate(interpolate((Y[:, t],S[:], ω[:], h_bar,), C_E_o[t, :, :, 2, :, :], Gridded(Linear())), Interpolations.Flat()) # Four-dimensional interpolation
        ao_extp_t_np = extrapolate(interpolate((Y[:, t],S[:], ω[:], h_bar,), A_E_o[t, :, :, 2, :, :], Gridded(Linear())), Interpolations.Flat()) # Four-dimensional interpolation
        vo_extp_t_np = extrapolate(interpolate((Y[:, t],S[:], ω[:], h_bar,), V_E_o[t, :, :, 2, :, :], Gridded(Linear())), Interpolations.Flat())  # Four-dimensional interpolation

        #### PARTICIPATING
        # Renters
        cr_extp_t_ap = extrapolate(interpolate((Y[:, t],S[:],), C_E_r[t, :, :, 1], Gridded(Linear())), Interpolations.Flat())  # Two-dimensional interpolation
        hr_extp_t_ap = extrapolate(interpolate((Y[:, t],S[:],), H_E_r[t, :, :, 1], Gridded(Linear())), Interpolations.Flat())  # Two-dimensional interpolation
        ar_extp_t_ap = extrapolate(interpolate((Y[:, t],S[:],), A_E_r[t, :, :, 1], Gridded(Linear())), Interpolations.Flat())  # Two-dimensional interpolation
        vr_extp_t_ap = extrapolate(interpolate((Y[:, t],S[:],), V_E_r[t, :, :, 1], Gridded(Linear())), Interpolations.Flat())  # Two-dimensional interpolation

        # purchasers
        cp_extp_t_ap = extrapolate(interpolate((Y[:, t],S[:],), C_E_p[t, :, :, 1], Gridded(Linear())), Interpolations.Flat())  # Two-dimensional interpolation
        hp_extp_t_ap = extrapolate(interpolate((Y[:, t],S[:],), H_E_p[t, :, :, 1], Gridded(Constant())), Interpolations.Flat())  # Two-dimensional interpolation
        vp_extp_t_ap = extrapolate(interpolate((Y[:, t],S[:],), V_E_p[t, :, :, 1], Gridded(Linear())), Interpolations.Flat())  # Two-dimensional interpolation
        ap_extp_t_ap = extrapolate(interpolate((Y[:, t],S[:],), A_E_p[t, :, :, 1], Gridded(Linear())), Interpolations.Flat())  # Two-dimensional interpolation

        # Owners
        co_extp_t_ap = extrapolate(interpolate((Y[:, t],S[:], ω[:], h_bar,), C_E_o[t, :, :, 1, :, :], Gridded(Linear())), Interpolations.Flat()) # Four-dimensional interpolation
        ao_extp_t_ap = extrapolate(interpolate((Y[:, t],S[:], ω[:], h_bar,), A_E_o[t, :, :, 1, :, :], Gridded(Linear())), Interpolations.Flat()) # Four-dimensional interpolation
        vo_extp_t_ap = extrapolate(interpolate((Y[:, t],S[:], ω[:], h_bar,), V_E_o[t, :, :, 1, :, :], Gridded(Linear())), Interpolations.Flat())  # Four-dimensional interpolation
        
        for i in 1:A
            if t == 1
                YT[i, t] = rand(LogNormal(lognormal_mu,lognormal_st), 1)[1] *12 /10000 # draw  a starting incomes from lognormal distribution
                if  YT[i, t] > findmax(Y[:,1])[1]
                    YT[i, t] = findmax(Y[:,1])[1]
                end
            else
                if t <= N
                    YT[i, t] = YT[i, t-1]*(1+rand(Normal(0, 0.1,), 1)[1])*ypsilon[t]
                else
                    YT[i, t] = YT[i, N]*z
                end

                RG[i, t] = (RT[t]+rand(Normal(0, 0.01,), 1)[1]-r_free)
                LT[i, t] =  ST[i, t-1]*(1+r_free) + AT[i, t-1]*ST[i, t-1]*RG[i, t]
                if LT[i, t] < 1/10000
                    LT[i, t] = 0 
                end

                if DT[i, t] == 1 && DP[i, t] != 1
                    OT[i, t] = (pt[t]/pt[t-1] - (1-OT[i, t-1]))/(pt[t]/pt[t-1])
                end
            end

            I_entry_t = I_entry*mean(Y[:, t])

            if  t == 1 || AP[i, t-1] == 0
                
                cr_extp_t = cr_extp_t_np
                hr_extp_t = hr_extp_t_np
                ar_extp_t = ar_extp_t_np 
                vr_extp_t = vr_extp_t_np

                # purchasers
                cp_extp_t = cp_extp_t_np
                hp_extp_t = hp_extp_t_np
                vp_extp_t = vp_extp_t_np
                ap_extp_t = ap_extp_t_np

                # Owners
                co_extp_t = co_extp_t_np
                ao_extp_t = ao_extp_t_np
                vo_extp_t = vo_extp_t_np
            else 
                # Renters
                cr_extp_t = cr_extp_t_ap
                hr_extp_t = hr_extp_t_ap
                ar_extp_t = ar_extp_t_ap
                vr_extp_t = vr_extp_t_ap

                # purchasers
                cp_extp_t = cp_extp_t_ap
                hp_extp_t = hp_extp_t_ap
                vp_extp_t = vp_extp_t_ap
                ap_extp_t = ap_extp_t_ap

                # Owners
                co_extp_t = co_extp_t_ap
                ao_extp_t = ao_extp_t_ap
                vo_extp_t = vo_extp_t_ap
            end     
    
            if DT[i, t] == 0 
                PR[i, t] = prob_d_f(vr_extp_t(YT[i, t], LT[i, t]), vp_extp_t(YT[i, t], LT[i, t]))
                if PR[i, t] == 1 && t < N-1 &&
                    hp_extp_t(YT[i, t], LT[i, t]) >=30  &&
                    YT[i, t] + LT[i, t] - cp_extp_t(YT[i, t], LT[i, t]) - hp_extp_t(YT[i, t], LT[i, t])*pt[t]*((ψ_r*(1-δ)+δ))  > 0
                    
                    nr += 1

                    h_pur = hp_extp_t(YT[i, t], LT[i, t])
                    c_pur = cp_extp_t(YT[i, t], LT[i, t])

                    CT[i, t] =  c_pur
                    HT[i, t] =  h_pur

                    VT[i, t] =  vp_extp_t(YT[i, t], LT[i, t])       
                    ST[i, t] = YT[i, t] + LT[i, t] -  CT[i, t] - HT[i, t]*pt[t]*((ψ_r*(1-δ)+δ))

                    if ST[i, t] < 1/10000 && ST[i, t] > -1/10000 # Round small values
                        ST[i, t] = 0 
                    end 

                    # Risky assets
                    at_tempt = ap_extp_t_ap(YT[i, t], ST[i, t])
                    if t > 1 && AP[i, t-1] == 1  # If already participating 
                        AT[i, t] =  ap_extp_t(YT[i, t], ST[i, t]) 
                        ST[i, t] = ST[i, t] - mean(YT[i, t]) * I_part * AT[i, t]
                        if AT[i, t] == 0 
                            AP[i, t] = 0
                        else 
                            AP[i, t] = 1
                        end
                    else 
                        at_tempt = ap_extp_t_ap(YT[i, t], ST[i, t])
                        if ST[i, t] - mean(YT[i, t]) * I_part * at_tempt - I_entry_t > 0 && vp_extp_t_ap(YT[i, t], ST[i, t]- mean(YT[i, t]) * I_part * at_tempt - I_entry_t) > vp_extp_t_np(YT[i, t], ST[i, t]) && at_tempt > 0
                            ST[i, t] = ST[i, t] - I_entry_t - mean(YT[i, t]) * I_part * AT[i, t]
                            AT[i, t] =  at_tempt
                            AP[i, t] = 1 
                        else 
                            AP[i, t] = 0
                        end
                    end

                    OT[i, t] = δ 
                    DT[i, t+1] = 1 
                    DP[i, t]  = 1
                    DT[i, t] = 1
                    PP[i, :] .= pt[t]

                else
                    CT[i, t] =  cr_extp_t(YT[i, t], LT[i, t])
                    HT[i, t] =  hr_extp_t(YT[i, t], LT[i, t])
                    VT[i, t] =  vr_extp_t(YT[i, t], LT[i, t])       
                    ST[i, t] = YT[i, t] + LT[i, t] -  CT[i, t] - HT[i, t]*pt[t]*ψ_r

                    if ST[i, t] < 1/10000 && ST[i, t] > -1/10000 
                        ST[i, t] = 0 
                    end 

                    at_tempt = ar_extp_t_ap(YT[i, t], ST[i, t])
                    if t > 1 && AP[i, t-1] == 1 # If already participating 
                        AT[i, t] =  ar_extp_t(YT[i, t], ST[i, t]) 
                        ST[i, t] = ST[i, t] - mean(YT[i, t]) * I_part * AT[i, t]
                        if AT[i, t] == 0 
                            AP[i, t] = 0
                        else 
                            AP[i, t] = 1
                        end
                    else 
                        at_tempt = ar_extp_t_ap(YT[i, t], ST[i, t])
                        if ST[i, t] - mean(YT[i, t]) * I_part * at_tempt - I_entry_t > 0 && vr_extp_t_ap(YT[i, t], ST[i, t]- mean(YT[i, t]) * I_part * at_tempt - I_entry_t) > vr_extp_t_np(YT[i, t], ST[i, t]) && at_tempt > 0
                            ST[i, t] = ST[i, t] - I_entry_t - mean(YT[i, t]) * I_part * AT[i, t]
                            AT[i, t] =  at_tempt
                            AP[i, t] = 1 
                        else 
                            AT[i, t] = 0 
                        end
                    end
                end

                if ST[i, t] < 0
                    DF[i, t] = 1 # Mark defaulting agents
                    CT[i, t] =  cr_extp_t(closest_value_y(t, YT[i, t]), closest_value_s(t, LT[i, t]))
                    HT[i, t] =  hr_extp_t(closest_value_y(t, YT[i, t]), closest_value_s(t, LT[i, t]))
                    VT[i, t]  = vr_extp_t(closest_value_y(t, YT[i, t]), closest_value_s(t, LT[i, t]))
                    ST[i, t] = 0
                end

            else 
                CT[i, t] =  co_extp_t(YT[i, t], LT[i, t], OT[i, t], HT[i, t])
                VT[i, t] =  vo_extp_t(YT[i, t], LT[i, t], OT[i, t], HT[i, t]) 
                HT[i, t] =  HT[i, t-1]      
                if t < T 
                    DT[i, t+1] = 1
                end 
                ST[i, t] = YT[i, t] + LT[i, t] -  CT[i, t] - HT[i, t]*(1 -OT[i, t])*pt[t] *ψ_o
                if ST[i, t] < 1/10000 && ST[i, t] > -1/10000 
                    ST[i, t] = 0 
                end 
                at_tempt =  ao_extp_t(YT[i, t], ST[i, t], OT[i, t], HT[i, t])
                if t > 1 && AP[i, t-1] == 1 # If already participating 
                    AT[i, t] =  ao_extp_t(YT[i, t], ST[i, t], OT[i, t], HT[i, t])
                    ST[i, t] = ST[i, t] - mean(YT[i, t]) * I_part * AT[i, t]
                    if AT[i, t] == 0 
                        AP[i, t] = 0
                    else 
                        AP[i, t] = 1
                    end
                else 
                    at_tempt =  ao_extp_t(YT[i, t], ST[i, t], OT[i, t], HT[i, t])
                    if ST[i, t] - mean(YT[i, t]) * I_part * at_tempt - I_entry_t > 0 && vo_extp_t_ap(YT[i, t], ST[i, t]- mean(YT[i, t]) * I_part * at_tempt - I_entry_t, OT[i, t], HT[i, t]) > vo_extp_t_np(YT[i, t], ST[i, t], OT[i, t], HT[i, t]) && at_tempt > 0
                        ST[i, t] = ST[i, t] - I_entry_t - mean(YT[i, t]) * I_part * AT[i, t]
                        AT[i, t] =  at_tempt
                        AP[i, t] = 1
                    else 
                        AT[i, t] = 0 
                    end
                end
                if ST[i, t] < 0 
                    DF[i, t] = 1
                    CT[i, t] =  YT[i, t] + LT[i, t] - HT[i, t]*pt[t]*((1-OT[i, t])*ψ_o) 
                    HT[i, t] =  HT[i, t-1]      
                    AT[i, t]  = 0 
                    AP[i, t] = 0
                    VT[i, t]  =  vo_extp_t(YT[i, t], LT[i, t], OT[i, t], HT[i, t]) 
                    ST[i, t] = 0 
                end
            end
        end
    end

    ####################################################################################################################
    # Stack data and save dataset
    ptmat.time = string.(ptmat.time)

    YT = stack(DataFrame(YT, :auto))
    rename!(YT, "variable" => "time", "value" => "YT")
    YT.time = replace.(YT.time, "x" => "")
    YT.row = rownumber.(eachrow(YT))

    CT = stack(DataFrame(CT, :auto))
    rename!(CT, "variable" => "time", "value" => "CT")
    select!(CT, Not(:time))
    CT.row = rownumber.(eachrow(CT))

    HT = stack(DataFrame(HT, :auto))
    rename!(HT, "variable" => "time", "value" => "HT")
    select!(HT, Not(:time))
    HT.row = rownumber.(eachrow(HT))

    ST = stack(DataFrame(ST, :auto))
    rename!(ST, "variable" => "time", "value" => "ST")
    select!(ST, Not(:time))
    ST.row = rownumber.(eachrow(ST))

    LT = stack(DataFrame(LT, :auto))
    rename!(LT, "variable" => "time", "value" => "LT")
    select!(LT, Not(:time))
    LT.row = rownumber.(eachrow(LT))

    AT = stack(DataFrame(AT, :auto))
    rename!(AT, "variable" => "time", "value" => "AT")
    select!(AT, Not(:time))
    AT.row = rownumber.(eachrow(AT))

    AP = stack(DataFrame(AP, :auto))
    rename!(AP, "variable" => "time", "value" => "AP")
    select!(AP, Not(:time))
    AP.row = rownumber.(eachrow(AP))

    VT = stack(DataFrame(VT, :auto))
    rename!(VT, "variable" => "time", "value" => "VT")
    select!(VT, Not(:time))
    VT.row = rownumber.(eachrow(VT))

    PR = stack(DataFrame(PR, :auto))
    rename!(PR, "variable" => "time", "value" => "PR")
    select!(PR, Not(:time))
    PR.row = rownumber.(eachrow(PR))

    OT = stack(DataFrame(OT, :auto))
    rename!(OT, "variable" => "time", "value" => "OT")
    select!(OT, Not(:time))
    OT.row = rownumber.(eachrow(OT))

    DT = stack(DataFrame(DT, :auto))
    rename!(DT, "variable" => "time", "value" => "DT")
    select!(DT, Not(:time))
    DT.row = rownumber.(eachrow(DT))

    DP = stack(DataFrame(DP, :auto))
    rename!(DP, "variable" => "time", "value" => "DP")
    select!(DP, Not(:time))
    DP.row = rownumber.(eachrow(DP))

    RG = stack(DataFrame(RG, :auto))
    rename!(RG, "variable" => "time", "value" => "RT")
    select!(RG, Not(:time))
    RG.row = rownumber.(eachrow(RG))

    PP = stack(DataFrame(PP, :auto))
    rename!(PP, "variable" => "time", "value" => "PP")
    select!(PP, Not(:time))
    PP.row = rownumber.(eachrow(PP))

    DF = stack(DataFrame(DF, :auto))
    rename!(DF, "variable" => "time", "value" => "DF")
    select!(DF, Not(:time))
    DF.row = rownumber.(eachrow(DF))

    df =  innerjoin(YT, CT, on = :row)
    df =  innerjoin(df, HT, on = :row)
    df =  innerjoin(df, ST, on = :row)
    df =  innerjoin(df, LT, on = :row)
    df =  innerjoin(df, AT, on = :row)
    df =  innerjoin(df, AP, on = :row)
    df =  innerjoin(df, VT, on = :row)
    df =  innerjoin(df, PR, on = :row)
    df =  innerjoin(df, OT, on = :row)
    df =  innerjoin(df, DT, on = :row)
    df =  innerjoin(df, DP, on = :row)
    df =  innerjoin(df, RG, on = :row)
    df =  innerjoin(df, PP, on = :row)
    df =  innerjoin(df, DF, on = :row)
    df = innerjoin(df, ptmat, on = :time)

    if save == true
        cd(results)
        CSV.write(raw"df_d.csv", df)
    end
    return(df)
end

function df_creation_contrafactual(agent_n, seed_n, save) # Function for generating counterfactual agents
    A = agent_n
    Random.seed!(seed_n)
    RT = rand(Normal(0.06, 0.16,), (T))
    ####################################################################################################################
    ### Create matrices
    YT = zeros(Float64,(A, T)) # create matrix of income profiles
    CT = zeros(Float64,(A, T)) # create matrix of housing profiles
    HT = zeros(Float64,(A, T)) # create matrix of housing profiles
    ST = zeros(Float64,(A, T)) # create matrix of liquid assets profiles
    LT = zeros(Float64,(A, T)) # create matrix of cash at hand profiles
    AT = zeros(Float64,(A, T)) # create matrix of risky shares profiles
    AP = zeros(Float64,(A, T)) # create matrix of participation
    VT = zeros(Float64,(A, T)) # create matrix of utility profiles
    PR = zeros(Float64,(A, T)) # create matrix of purchasing probability
    OT = zeros(Float64,(A, T)) # create matrix of mortgages
    DT = zeros(Float64,(A, T)) # create matrix for ownership
    DP = zeros(Float64,(A, T)) # create matrix for purchase
    RG = zeros(Float64,(A, T)) # create matrix for stock returns
    DF = zeros(Float64,(A, T)) # create matrix for default
    ####################################################################################################################
    # Life-Cycle simulation
    # Workers
    for t in 1:T
        #println(t)
        ## Interpolations
        #### NON PARTICIPATING
        # Renters
        cr_extp_t_np = extrapolate(interpolate((Y[:, t],S[:],), C_E_r[t, :, :, 2], Gridded(Linear())), Interpolations.Flat())  # Two-dimensional interpolation
        hr_extp_t_np = extrapolate(interpolate((Y[:, t],S[:],), H_E_r[t, :, :, 2], Gridded(Linear())), Interpolations.Flat())  # Two-dimensional interpolation
        ar_extp_t_np = extrapolate(interpolate((Y[:, t],S[:],), A_E_r[t, :, :, 2], Gridded(Linear())), Interpolations.Flat())  # Two-dimensional interpolation
        vr_extp_t_np = extrapolate(interpolate((Y[:, t],S[:],), V_E_r[t, :, :, 2], Gridded(Linear())), Interpolations.Flat())  # Two-dimensional interpolation

        # Counterfactual Renters
        ccr_extp_t_np = extrapolate(interpolate((Y[:, t],S[:],), C_E_cr[t, :, :, 2], Gridded(Linear())), Interpolations.Flat())  # Two-dimensional interpolation
        hcr_extp_t_np = extrapolate(interpolate((Y[:, t],S[:],), H_E_cr[t, :, :, 2], Gridded(Linear())), Interpolations.Flat())  # Two-dimensional interpolation
        acr_extp_t_np = extrapolate(interpolate((Y[:, t],S[:],), A_E_cr[t, :, :, 2], Gridded(Linear())), Interpolations.Flat())  # Two-dimensional interpolation
        vcr_extp_t_np = extrapolate(interpolate((Y[:, t],S[:],), V_E_cr[t, :, :, 2], Gridded(Linear())), Interpolations.Flat())  # Two-dimensional interpolation
        
        # purchasers
        cp_extp_t_np = extrapolate(interpolate((Y[:, t],S[:],), C_E_p[t, :, :, 2], Gridded(Linear())), Interpolations.Flat())  # Two-dimensional interpolation
        hp_extp_t_np = extrapolate(interpolate((Y[:, t],S[:],), H_E_p[t, :, :, 2], Gridded(Constant())), Interpolations.Flat())  # Two-dimensional interpolation
        vp_extp_t_np = extrapolate(interpolate((Y[:, t],S[:],), V_E_p[t, :, :, 2], Gridded(Linear())), Interpolations.Flat())  # Two-dimensional interpolation

        #### PARTICIPATING
        # Renters
        cr_extp_t_ap = extrapolate(interpolate((Y[:, t],S[:],), C_E_r[t, :, :, 1], Gridded(Linear())), Interpolations.Flat())  # Two-dimensional interpolation
        hr_extp_t_ap = extrapolate(interpolate((Y[:, t],S[:],), H_E_r[t, :, :, 1], Gridded(Linear())), Interpolations.Flat())  # Two-dimensional interpolation
        ar_extp_t_ap = extrapolate(interpolate((Y[:, t],S[:],), A_E_r[t, :, :, 1], Gridded(Linear())), Interpolations.Flat())  # Two-dimensional interpolation
        vr_extp_t_ap = extrapolate(interpolate((Y[:, t],S[:],), V_E_r[t, :, :, 1], Gridded(Linear())), Interpolations.Flat())  # Two-dimensional interpolation
        
        # Counterfactual Renters
        ccr_extp_t_ap = extrapolate(interpolate((Y[:, t],S[:],), C_E_cr[t, :, :, 1], Gridded(Linear())), Interpolations.Flat())  # Two-dimensional interpolation
        hcr_extp_t_ap = extrapolate(interpolate((Y[:, t],S[:],), H_E_cr[t, :, :, 1], Gridded(Linear())), Interpolations.Flat())  # Two-dimensional interpolation
        acr_extp_t_ap = extrapolate(interpolate((Y[:, t],S[:],), A_E_cr[t, :, :, 1], Gridded(Linear())), Interpolations.Flat())  # Two-dimensional interpolation
        vcr_extp_t_ap = extrapolate(interpolate((Y[:, t],S[:],), V_E_cr[t, :, :, 1], Gridded(Linear())), Interpolations.Flat())  # Two-dimensional interpolation

        # purchasers
        cp_extp_t_ap = extrapolate(interpolate((Y[:, t],S[:],), C_E_p[t, :, :, 1], Gridded(Linear())), Interpolations.Flat())  # Two-dimensional interpolation
        hp_extp_t_ap = extrapolate(interpolate((Y[:, t],S[:],), H_E_p[t, :, :, 1], Gridded(Constant())), Interpolations.Flat())  # Two-dimensional interpolation
        vp_extp_t_ap = extrapolate(interpolate((Y[:, t],S[:],), V_E_p[t, :, :, 1], Gridded(Linear())), Interpolations.Flat())  # Two-dimensional interpolation
        
        for i in 1:A
            if t == 1
                YT[i, t] = rand(LogNormal(lognormal_mu,lognormal_st), 1)[1] *12 /10000 # draw  a starting incomes from lognormal distribution
                if  YT[i, t] > findmax(Y[:,1])[1]
                    YT[i, t] = findmax(Y[:,1])[1]
                end
            else
                if t <= N
                    YT[i, t] = YT[i, t-1]*(1+rand(Normal(0, 0.1,), 1)[1])*ypsilon[t]
                else
                    YT[i, t] = YT[i, N]*z
                end

                RG[i, t] = (RT[t]+rand(Normal(0, 0.01,), 1)[1]-r_free)
                LT[i, t] =  ST[i, t-1]*(1+r_free) + AT[i, t-1]*ST[i, t-1]*RG[i, t]
                if LT[i, t] < 1/10000
                    LT[i, t] = 0 
                end
            end

            I_entry_t = I_entry*mean(Y[:, t])

            if  t == 1 || AP[i, t-1] == 0
                
                # Renters
                cr_extp_t = cr_extp_t_np
                hr_extp_t = hr_extp_t_np
                ar_extp_t = ar_extp_t_np 
                vr_extp_t = vr_extp_t_np

                # purchasers
                cp_extp_t = cp_extp_t_np
                hp_extp_t = hp_extp_t_np
                vp_extp_t = vp_extp_t_np
                
                # Counterfactual owners
                ccr_extp_t = ccr_extp_t_np
                hcr_extp_t = hcr_extp_t_np
                acr_extp_t = acr_extp_t_np 
                vcr_extp_t = vcr_extp_t_np

            else 
                # Renters
                cr_extp_t = cr_extp_t_ap
                hr_extp_t = hr_extp_t_ap
                ar_extp_t = ar_extp_t_ap
                vr_extp_t = vr_extp_t_ap

                # purchasers
                cp_extp_t = cp_extp_t_ap
                hp_extp_t = hp_extp_t_ap
                vp_extp_t = vp_extp_t_ap

                # Counterfactual owners
                ccr_extp_t = ccr_extp_t_ap
                hcr_extp_t = hcr_extp_t_ap
                acr_extp_t = acr_extp_t_ap 
                vcr_extp_t = vcr_extp_t_ap
            end  
            
                
            if DT[i, t] == 0 
                PR[i, t] = prob_d_f(vr_extp_t(YT[i, t], LT[i, t]), vp_extp_t(YT[i, t], LT[i, t]))
                if PR[i, t] == 1 && t < N-1 &&
                    hp_extp_t(YT[i, t], LT[i, t]) >=30  &&
                    YT[i, t] + LT[i, t] - cp_extp_t(YT[i, t], LT[i, t]) - hp_extp_t(YT[i, t], LT[i, t])*pt[t]*((ψ_r*(1-δ)+δ)) > 0
                    OT[i, :] .= t
                    DT[i, t:T] .= 1 
                    DP[i, t]  = 1
                end
            end

            if DT[i, t] == 0 

                CT[i, t] =  cr_extp_t(YT[i, t], LT[i, t])
                HT[i, t] =  hr_extp_t(YT[i, t], LT[i, t])
                VT[i, t] =  vr_extp_t(YT[i, t], LT[i, t])       
                ST[i, t] = YT[i, t] + LT[i, t] -  CT[i, t] - HT[i, t]*pt[t]*ψ_r

                if ST[i, t] < 1/10000 && ST[i, t] > -1/10000 
                    ST[i, t] = 0 
                end 

                at_tempt = ar_extp_t_ap(YT[i, t], ST[i, t])
                if t > 1 && AP[i, t-1] == 1  #&& ST[i, t] - mean(YT[i, t]) * I_part * at_tempt > 0 && vr_extp_t_ap(YT[i, t], ST[i, t]- mean(YT[i, t]) * I_part * at_tempt) > vr_extp_t_np(YT[i, t], ST[i, t]) && at_tempt > 0 # If already participating 
                    AT[i, t] =  ar_extp_t(YT[i, t], ST[i, t]) 
                    ST[i, t] = ST[i, t] - mean(YT[i, t]) * I_part * AT[i, t]
                    if AT[i, t] == 0 
                        AP[i, t] = 0
                    else 
                        AP[i, t] = 1
                    end
                else 
                    at_tempt = ar_extp_t_ap(YT[i, t], ST[i, t])
                    if ST[i, t] - mean(YT[i, t]) * I_part * at_tempt - I_entry_t > 0 && vr_extp_t_ap(YT[i, t], ST[i, t]- mean(YT[i, t]) * I_part * at_tempt - I_entry_t) > vr_extp_t_np(YT[i, t], ST[i, t]) && at_tempt > 0
                        ST[i, t] = ST[i, t] - I_entry_t - mean(YT[i, t]) * I_part * AT[i, t]
                        AT[i, t] =  at_tempt
                        AP[i, t] = 1
                    else 
                        AT[i, t] = 0 
                    end
                end   
            else 

                CT[i, t] =  ccr_extp_t(YT[i, t], LT[i, t])
                HT[i, t] =  hcr_extp_t(YT[i, t], LT[i, t])
                VT[i, t] =  vcr_extp_t(YT[i, t], LT[i, t])       
                ST[i, t] = YT[i, t] + LT[i, t] -  CT[i, t] - HT[i, t]*pt[t]*ψ_r

                if ST[i, t] < 1/10000 && ST[i, t] > -1/10000 
                    ST[i, t] = 0 
                end 

                at_tempt = acr_extp_t_ap(YT[i, t], ST[i, t])
                if t > 1 && AP[i, t-1] == 1 # If already participating 
                    AT[i, t] =  acr_extp_t(YT[i, t], ST[i, t]) 
                    ST[i, t] = ST[i, t] - mean(YT[i, t]) * I_part * AT[i, t]
                    if AT[i, t] == 0 
                        AP[i, t] = 0
                    else 
                        AP[i, t] = 1
                    end
                else 
                    at_tempt = acr_extp_t_ap(YT[i, t], ST[i, t])
                    if ST[i, t] - mean(YT[i, t]) * I_part * at_tempt - I_entry_t > 0 && vcr_extp_t_ap(YT[i, t], ST[i, t]- mean(YT[i, t]) * I_part * at_tempt - I_entry_t) > vcr_extp_t_np(YT[i, t], ST[i, t]) && at_tempt > 0
                        ST[i, t] = ST[i, t] - I_entry_t - mean(YT[i, t]) * I_part * AT[i, t]
                        AT[i, t] =  at_tempt
                        AP[i, t] = 1
                    else 
                        AT[i, t] = 0 
                    end
                end
            end
        end
    end

    ####################################################################################################################
    # Stack data and save dataset
    ptmat.time = string.(ptmat.time)

    YT = stack(DataFrame(YT, :auto))
    rename!(YT, "variable" => "time", "value" => "YT")
    YT.time = replace.(YT.time, "x" => "")
    YT.row = rownumber.(eachrow(YT))

    CT = stack(DataFrame(CT, :auto))
    rename!(CT, "variable" => "time", "value" => "CT")
    select!(CT, Not(:time))
    CT.row = rownumber.(eachrow(CT))

    HT = stack(DataFrame(HT, :auto))
    rename!(HT, "variable" => "time", "value" => "HT")
    select!(HT, Not(:time))
    HT.row = rownumber.(eachrow(HT))

    ST = stack(DataFrame(ST, :auto))
    rename!(ST, "variable" => "time", "value" => "ST")
    select!(ST, Not(:time))
    ST.row = rownumber.(eachrow(ST))

    LT = stack(DataFrame(LT, :auto))
    rename!(LT, "variable" => "time", "value" => "LT")
    select!(LT, Not(:time))
    LT.row = rownumber.(eachrow(LT))

    AT = stack(DataFrame(AT, :auto))
    rename!(AT, "variable" => "time", "value" => "AT")
    select!(AT, Not(:time))
    AT.row = rownumber.(eachrow(AT))

    AP = stack(DataFrame(AP, :auto))
    rename!(AP, "variable" => "time", "value" => "AP")
    select!(AP, Not(:time))
    AP.row = rownumber.(eachrow(AP))

    VT = stack(DataFrame(VT, :auto))
    rename!(VT, "variable" => "time", "value" => "VT")
    select!(VT, Not(:time))
    VT.row = rownumber.(eachrow(VT))

    PR = stack(DataFrame(PR, :auto))
    rename!(PR, "variable" => "time", "value" => "PR")
    select!(PR, Not(:time))
    PR.row = rownumber.(eachrow(PR))

    OT = stack(DataFrame(OT, :auto))
    rename!(OT, "variable" => "time", "value" => "OT")
    select!(OT, Not(:time))
    OT.row = rownumber.(eachrow(OT))

    DT = stack(DataFrame(DT, :auto))
    rename!(DT, "variable" => "time", "value" => "DT")
    select!(DT, Not(:time))
    DT.row = rownumber.(eachrow(DT))

    DP = stack(DataFrame(DP, :auto))
    rename!(DP, "variable" => "time", "value" => "DP")
    select!(DP, Not(:time))
    DP.row = rownumber.(eachrow(DP))

    RG = stack(DataFrame(RG, :auto))
    rename!(RG, "variable" => "time", "value" => "RT")
    select!(RG, Not(:time))
    RG.row = rownumber.(eachrow(RG))

    DF = stack(DataFrame(DF, :auto))
    rename!(DF, "variable" => "time", "value" => "DF")
    select!(DF, Not(:time))
    DF.row = rownumber.(eachrow(DF))

    df =  innerjoin(YT, CT, on = :row)
    df =  innerjoin(df, HT, on = :row)
    df =  innerjoin(df, ST, on = :row)
    df =  innerjoin(df, LT, on = :row)
    df =  innerjoin(df, AT, on = :row)
    df =  innerjoin(df, AP, on = :row)
    df =  innerjoin(df, VT, on = :row)
    df =  innerjoin(df, PR, on = :row)
    df =  innerjoin(df, OT, on = :row)
    df =  innerjoin(df, DT, on = :row)
    df =  innerjoin(df, DP, on = :row)
    df =  innerjoin(df, RG, on = :row)
    df =  innerjoin(df, DF, on = :row)
    df = innerjoin(df, ptmat, on = :time)

    if save == true
        cd(results)
        CSV.write(raw"df_contrafactual_d.csv", df)
    end
    return(df)
end

function data_prep(df) # Prepare dataframe for analysis
    # Generate an id by grouping by `time` and using row number within each group
    transform!(groupby(df, [:time]), :time => eachindex => :id)

    # Pull id to the fron 
    select!(df, [:id, :time], :)

    # Sort by id and time
    sort!(df, [:id, :time])

    # Create flag for (eventual) owner
    transform!(groupby(df, [:id]), :DT => maximum => :owner)

    # Create variable for total wealth 
    df.time .= parse.(Int64, df.time)
    df.purchase = ifelse.(df.DP .== 1, df.time, 0)
    transform!(groupby(df, [:id]), :purchase => maximum => :purchase)

    # Sort by id and time
    sort!(df, [:id, :time])

    return(df)
end 

###########################################################################################################
#### GENERATE DATA
begin # Load data if not loaded already
    cd(results)
    C_E_o = load("C_E_o.jld", "data")
    A_E_o = load("A_E_o.jld", "data")
    V_E_o = load("V_E_o.jld", "data")
    C_E_r = load("C_E_r.jld", "data")
    A_E_r = load("A_E_r.jld", "data")
    V_E_r = load("V_E_r.jld", "data")
    H_E_r = load("H_E_r.jld", "data")
    C_E_p = load("C_E_p.jld", "data")
    A_E_p = load("A_E_p.jld", "data")
    V_E_p = load("V_E_p.jld", "data")
    H_E_p = load("H_E_p.jld", "data")
    C_E_cr = load("C_E_cr.jld", "data")
    A_E_cr = load("A_E_cr.jld", "data")
    V_E_cr = load("V_E_cr.jld", "data")
    H_E_cr = load("H_E_cr.jld", "data")

    println("loaded")
end 


df_or = df_creation(10000, 10000, false)
df_or_cf= df_creation_contrafactual(10000, 10000, false)

df_baseline = data_prep(df_or)
df_counterfactual  = data_prep(df_or_cf)


####################################################################################################################
### Write winzorization function

function winsor2(var, prop)
    per = percentile.(eachcol(var), prop)
    var = ifelse.(var .> per, per, var)
    return(var)
end

######################################################################
### Append dataframes

df_baseline.counterfactual .= 0 
df_counterfactual.counterfactual .= 1
df_counterfactual.id .= df_counterfactual.id .+ 10000

df_baseline.tot_wealth = df_baseline.ST .+ df_baseline.DT .* df_baseline.HT .* df_baseline.OT .* df_baseline.pt
df_counterfactual.tot_wealth = df_counterfactual.ST
df_counterfactual.PP .= 0

df_baseline.h_costs = df_baseline.HT .* df_baseline.pt .* (ψ_r + 0.2) .* df_baseline.DP .* df_baseline.DT + df_baseline.HT .* df_baseline.pt .* (1 .- df_baseline.OT) .* (ψ_r) .* (1 .- df_baseline.DP) .* df_baseline.DT + df_baseline.HT .* df_baseline.pt .* (ψ_r) .* (1 .- df_baseline.DT)
df_counterfactual.h_costs = df_counterfactual.HT .* df_counterfactual.pt .* (ψ_r)

df = vcat(df_baseline,df_counterfactual) 

######################################################################
### Convert prices to 2020 CHF
cd(background_data)
index_df = CSV.read("index.csv", DataFrame)
index_df.time .= index_df.year .- 1977
select!(index_df, [:time, :year, :index_forecast])
leftjoin!(df, index_df, on = :time) 

df.ST .= df.ST ./ df.index_forecast
df.tot_wealth .= df.tot_wealth ./ df.index_forecast
df.h_costs .= df.h_costs ./ df.index_forecast
df.CT .= df.CT ./ df.index_forecast
df.pt .= df.pt ./ df.index_forecast
df.YT .= df.YT ./ df.index_forecast
df.LT .= df.LT ./ df.index_forecast

df.tot_consumption .= (df.CT .+ df.h_costs)

df.ST = winsor2(df.ST/10, 95)
df.tot_wealth = winsor2(df.tot_wealth/10, 95)
df.h_costs =  winsor2(df.h_costs/10, 95)
df.CT =  winsor2(df.CT/10, 95)
df.pt =  df.pt
df.tot_consumption .= winsor2(df.tot_consumption/10, 95)
df.YT = df.YT/10
df.LT = winsor2(df.LT/10, 95)


######################################################################
### SAVE FINAL DATA

cd(results)
CSV.write("dataready.csv", df)

