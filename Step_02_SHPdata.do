global raw_shp "~\SHP\data\Data_STATA\Data_STATA" /* You need to specify your location of the raw SHP data in STATA format*/
global path_shp "~\background_data\shp" 
global background_data "~\background_data"

foreach i in 12 16 20 { 
	cd $raw_shp
	local k = `i' + 2
	use "SHP-Data-W1-W24-STATA\W`k'_20`i'\shp`i'_p_user.dta", clear // open individual file
	merge m:1 idhous`i' using "SHP-Data-W1-W24-STATA\W`k'_20`i'\shp`i'_h_user", nogen // add all variables from household file

	keep age`i' h`i'i111*  h`i'i110* nbadul* idpers h`i'h37 h`i'i54 h`i'i39 idhous`i' i`i'ptotn h`i'h06 h`i'i57

	keep if age`i' >= 25 & age`i' <= 95

	rename age`i' age 
	cap rename h`i'i111a hi111a
	cap rename h`i'i111b hi111b
	cap rename h`i'i111c hi111c
	cap rename h`i'i111d hi111d
	cap rename h`i'i110a hi110a
	cap rename h`i'i110b hi110b
	cap rename h`i'i110c hi110c
	cap rename h`i'i110d hi110d
	cap rename h`i'i57 ad_contrs
	cap rename nbadul`i' nr_adults
	cap rename h`i'h37 hh37
	cap rename h`i'i54 hi54
	cap rename idhous`i' idhous
	cap rename i`i'ptotn iptotn
	cap rename h`i'h06 hh06
	cap rename h`i'i39 hi39

	gen wave = `i'

	cd $path_shp
	save "wave`i'", replace
} 

cd $path_shp
use wave12, clear
append using wave16
append using wave20

gen year = 2000 + wave + 2
cd $raw_shp
merge m:1 idhous year using "SHP-Data-Imputed-Income-Wealth-STATA\imputed_income_hh_long_shp", keep (1 3) nogen keepusing (ihtyni)
rename ihtyni income

replace hi110c = hi110a if hi110c == . & hi110a != . 
drop hi110a
replace hi111a = hi111c  if (hi111a < 0 & hi111c > 0) 
drop hi111c

drop hi110b hi111b hi111d
rename hi110c property 
rename hi111a assets
rename hi110d mortgage
rename hh37 housing_exp
rename hi54 min_income
rename iptotn income_pers
rename hh06 year_move
rename hi39 payments

gen owner = 1 if property > 0 
replace owner = 0 if property == -3
drop if owner == . 

replace assets = . if assets < 0 

replace housing_exp = 0 if housing_exp == -3
replace housing_exp = . if housing_exp < 0 

replace income_pers = 0 if income_pers == -3
replace income_pers = . if income_pers < 0

replace property = 0 if property == -3
replace property = . if property < 0 

order idpers wave age nr_adults owner 

gen assets_pc =  assets/nr_adults

cd $path_shp
save temp, replace

cd $background_data
import delimited "index.csv", clear

gen time = year - 1977
drop index

cd $path_shp
merge 1:m year using temp, keep(3) nogen

keep if income_pers > 0 

replace assets_pc  = assets_pc/index_forecast/100000
gen tot_costs = housing_exp*12/nr_adults/index_forecast/100000
replace income_pers = income_pers/index_forecast/100000
replace property = property/nr_adults/index_forecast/100000

winsor2 assets_pc income_pers tot_costs property, replace cuts(0 95)

rename assets_pc ST 
rename idpers id
rename income_pers YT
gen data = "shp"
gen DT = owner 
gen tot_wealth = ST + property 

keep id time age owner ST YT DT property tot_wealth tot_costs data

preserve
collapse (mean) ST (mean) YT (mean) property (mean) tot_wealth (mean) tot_costs (mean) DT, by(age)

sort age

gen data = "shp"

cd $background_data
export delimited using "shp.csv", replace
restore 


preserve
keep if DT == 1
collapse (mean) ST (mean) YT (mean) property (mean) tot_wealth (mean) tot_costs (mean) DT, by(age)

sort age

gen data = "shp"

cd $background_data
export delimited using "shp_owner.csv", replace
restore 



