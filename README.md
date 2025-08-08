# Horse Pedigree Prediction using Machine learning

## Label Overview (Before Data Preprocessing)
- Name: Name of the horse
- Form: Recent racing form/results
- Rating: official racing rating
- Raw Erg: measured speed/performance metric
- Erg: normalized speed/ performance metric **This is what we want to optimize**
- Ems: **unsure**
- Grade: class level of races
- YOB: birth year
- sex: gender
- sire: the father of the horse
- Fee: **cost to breed the horse?**
- Crop: **unsure**
- dam: the mother of the horse
- bmSire: the father of the dam (mother)
- form2: 2nd lastest form/result
- ems3: **unsure**
- grade4: **unsure**
- price: Sale price
- status: sale Status
- code: sale code
- lot: lot number at auction
- vendor: seller
- purhcaser: buyer
- prev.price: previous sale price

## Machine Model and Training
**Goal**: The goal is to create a model that can optimize the predicted erg of a horse to help determine the value of the horse. Possible models include:
- Gradient Boosting
- Random Forest
- TabNet