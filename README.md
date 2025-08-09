# Horse Pedigree Prediction using Machine learning

## Label Overview (Before Data Preprocessing)
- Name: Name of the horse
- Form: Recent racing form/results
- Rating: official racing rating
- Raw Erg: measured speed/performance metric
- Erg: Expected racehorse grade 
- Ems: Expected mane score (cut)
- Grade: class level of races (cut)
- YOB: birth year
- sex: gender
- sire: the father of the horse
- Fee: Cost to breed the horse
- Crop: number of years the horse had babies
- dam: the mother of the horse
- bmSire: the father of the dam (mother)
- form2: Mom's form
- ems3: score for the likelihood of the horses mom to produce a good horse
- grade4: (Cut)
- price: Sale price
- status: sale Status
- code: sale code
- lot: lot number at auction
- vendor: seller
- purhcaser: buyer
- prev.price: previous sale price

## How to run:
1. python must be installed [here](https://www.python.org/downloads/)
2. Create a virtual enviorment\
`python -m venv venv`\
 or  
`python3 -m venv venv`
3. Then start the enviorment:

    on Mac/Linux:\
    `source venv/bin/activate`

    on Windows:\
    `venv\Scripts\activate`

4. In the enviorment, run the following command: `pip install -r requirements.txt`

5. Then run the following code in `horsePrediction.ipynb`

## Machine Model and Training
**Goal**: The goal is to create a model that can optimize the predicted erg of a horse to help determine the value of the horse. Possible models include:
- Gradient Boosting
- Random Forest
- TabNet