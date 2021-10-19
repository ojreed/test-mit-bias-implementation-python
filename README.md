# test-mit-bias-implementation-python
Test implementation for MIT article implementation

Current issues:
    - tensorflow can't store in binary because I used an out of date implementation
    - AI manager cant run more than 1 dataset at a time because of an issue with the older tensorflow methodology i used
        - in the future I would use ts.sessions
Usage
    - change integer constant in the 3rd input assoicated of class call in line 178 to the dataset you want to run.
    - the second integer constant allows for changing which steps of data process we skip using the pickel library