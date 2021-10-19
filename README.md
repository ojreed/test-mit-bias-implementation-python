# test-mit-bias-implementation-python
Test implementation for MIT article implementation

Current issues:
    - tensorflow can't store in binary because I used an out of date training implementation
        - this means we just print out results to text after training 
        - not too much of an issue since the models train quickly on an individual level
    - AI manager cant run more than 1 dataset at a time because of an issue with the older tensorflow methodology i used
        - in the future I would use ts.sessions
        - it basicly cant handle the existance of multiple different size ANNs existing of the same type
            - would be an issue for large scale implementation but since this is really just a way to get familiar with some tools it is not as critical
Usage
    - change integer constant in the 3rd input assoicated of class call in line 178 to the dataset you want to run.
    - the second integer constant allows for changing which steps of data process we skip using the pickel library
    - this will print out the dataset to a txt file that includes a representation of the prbability each term will be a part of each article