global results "~\results"

********************************************************************************
* Determinants of wealth 

cd $results

import delimited dataready.csv, clear

gen intitial_income = yt if time == 1
bys id: ereplace intitial_income = max(intitial_income)

bys id (time): gen yt1 = yt[_n-1]
gen incomegr = (yt/yt1 -1)
bys id (time): gen incomegr_life = sum(incomegr) 

replace tot_wealth = ln(tot_wealth*100000)
replace st = ln(st*100000)
replace yt = ln(yt*100000)
replace intitial_income = ln(intitial_income*100000)

qui eststo st_40_bare: reg st intitial_income counterfactual if time == 40, vce(robust)
qui eststo st_40_mid: reg st intitial_income incomegr_life counterfactual if time == 40, vce(robust)
qui eststo st_40_full: reg st incomegr_life intitial_income dt##counterfactual ap  if time == 40, vce(robust)
qui eststo st_70_full: reg st incomegr_life intitial_income dt##counterfactual ap  if time == 70, vce(robust)

qui eststo w_40_bare: reg tot_wealth intitial_income counterfactual if time == 40, vce(robust)
qui eststo w_40_mid: reg tot_wealth intitial_income incomegr_life counterfactual if time == 40, vce(robust)
qui eststo w_40_full: reg tot_wealth incomegr_life intitial_income dt##counterfactual ap  if time == 40, vce(robust)
qui eststo w_70_full: reg tot_wealth incomegr_life intitial_income dt##counterfactual ap  if time == 70, vce(robust)


esttab st_40_bare st_40_mid st_40_full st_70_full w_40_bare w_40_mid w_40_full w_70_full, tex ///
b se(3) ///
keep(incomegr_life intitial_income  counterfactual 1.counterfactual 1.dt 1.dt#1.counterfactual  ap _cons) ///
varlabels(incomegr_life "Cum. income growth" intitial_income "\$\ln{Y_{1}}\$" counterfactual "Counterfactual" 1.counterfactual "Counterfactual"  1.dt "\$D_{65}\$" 1.dt#1.counterfactual "\$D_{65}\$ x Counterfactual" ap "\$A_{age}\$" _cons "Const.") ///
	mtitle( "\$\ln{S_{65}}\$" "\$\ln{S_{95}}\$" "\$\ln{S_{age}}\$" "\$\ln{W_{65}}\$" "\$\ln{W_{95}}\$" "\$\ln{W_{age}}\$") nogaps compress /// 
	addnotes("Note: Values are in terms of 2020 Swiss Francs. Values are winsorized at the bottom and top 5\%.") collabels(,none) substitute("\_" "_") ///
	s(r2 N, label( "R-sq" "N" ))
	

********************************************************************************
* Determinants of total consumption 

cd $results

import delimited dataready.csv, clear

gen event = time - purchase

gen lnct = ln(tot_consumption)
gen lnyt = ln(yt)

gen wh = ot * ht * dt * pt/10
replace wh = 0 if counterfactual == 1

gen initialwealth = lt + wh
gen lnlh = ln(initialwealth)

gen retired = time > 40 

qui eststo ct1: reg lnct lnyt lnlh counterfactual time if owner == 1 & event == 1, vce(robust)

qui eststo ct15: reg lnct lnyt lnlh counterfactual time if owner == 1 & event == 15, vce(robust)

qui eststo ct30: reg lnct lnyt lnlh counterfactual time if owner == 1 & event == 30, vce(robust)


esttab ct1 ct15 ct30, tex ///
b se(3) ///
varlabels(lnyt "\$\ln{Y_{e}}$\" counterfactual "Counterfactual" lnlh "\$\ln{(L_{e} + \omega_tP_t\overline{h}(1-Counterfactual))}$\" time "\$t_{e}\$" _cons "Const.") ///
	mtitle( "\$\ln{C_{1} + H_{1}}\$" "\$\ln{C_{15} + H_{15}}\$" "\$\ln{C_{30} + H_{30}}\$") nogaps compress /// 
	addnotes("Note: Monetary values are in terms of 2020 Swiss Francs. Values are winsorized at the bottom and top 5\%.") collabels(,none) substitute("\_" "_") ///
	s(r2 N, label( "R-sq" "N" ))

