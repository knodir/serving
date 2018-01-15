This folder contains changes made by Nodir et al.

We fork off version 1.4.0 (the latest release as of now, Jan. 13, 2018) to create the `dev` branch:
`git checkout tags/1.4.0 -b dev`. All of our changes will be done on the dev branch.
The `master` branch will track the upstream `master` (to pull the latest changes).
Here is how your branches should like like.
```
$ git remote --verbose
origin  https://github.com/knodir/serving.git (fetch)
origin  https://github.com/knodir/serving.git (push)
upstream        git@github.com:tensorflow/serving.git (fetch)
upstream        git@github.com:tensorflow/serving.git (push)
```

To make changes create a branch from the `dev` and push back to the `dev` branch (not `master`).
```
git checkout -b new-branch dev
# make your changes
git push origin dev
```
