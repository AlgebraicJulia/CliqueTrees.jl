# AlgebraicTemplate.jl

[![Stable Documentation](https://img.shields.io/badge/docs-stable-blue.svg)](https://AlgebraicJulia.github.io/AlgebraicTemplate.jl/stable)
[![Development Documentation](https://img.shields.io/badge/docs-dev-blue.svg)](https://AlgebraicJulia.github.io/AlgebraicTemplate.jl/dev)
[![Code Coverage](https://codecov.io/gh/AlgebraicJulia/AlgebraicTemplate.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/AlgebraicJulia/AlgebraicTemplate.jl)
[![CI/CD](https://github.com/AlgebraicJulia/AlgebraicTemplate.jl/actions/workflows/julia_ci.yml/badge.svg)](https://github.com/AlgebraicJulia/AlgebraicTemplate.jl/actions/workflows/julia_ci.yml)

A template repository for making a new AlgebraicJulia package.

## 🛠️ Usage

1. Use the "Use this template" dropdown to select "Create a new repository"
2. In the new page select "AlgebraicJulia" as the owner, give the repository a name, and create a new repository from the template
3. Set up Codecov credentials for code coverage (If you have trouble, reach out to an AlgebraicJulia organization owner to help with this)

   1. Log into [Codecov](https://codecov.io) with your GitHub account (this requires that you are a member of the AlgebraicJulia organization)
   2. Navigate to the [AlgebraicJulia organization](https://app.codecov.io/gh/AlgebraicJulia)
   3. Select your new repository from the list (e.x. "AlgebraicX")
   4. Note down the `CODECOV_TOKEN` value (It may be in the "Settings" tab if it doesn't show up immediately)
   5. Navigate back to your new GitHub repository and go to the Settings tab
   6. Go to "Security", "Secrets and variables", and "Actions" and click the "New repository secret" button
   7. Give the secret name `CODECOV_TOKEN` and the Secret value is the value you noted from the Codecov settings
   8. Click "Add secret"

4. Clone the new repository, for example in the terminal:
   ```sh
   git clone https://github.com/AlgebraicJulia/AlgebraicX.jl.git
   cd AlgebraicX.jl
   ```
5. Inspect for yourself and run `init.sh` with the new repository name and (optional) UUID are parameters. This script will substitute all instances of `AlgebraicX` with your new repository name and the default UUID with a new one or, if available, the UUID provided.
6. Go back to your repository and wait until the tests have passed, you can check the status by going to the "Actions" tab in the repository

### Buildkite

AlgebraicJulia uses [Buildkite](https://buildkite.com/) to submit resource-intensive processes such as building documentation and executing tests to the [HiPerGator](https://www.rc.ufl.edu/about/hipergator/) computing cluster.

While this template comes with a preconfigured `.buildkite/pipeline.yml` file, this repository is not integrated with Buildkite by default. If you would like your repository to use Buildkite to run processes on HiPerGator, tag an issue with @AlgebraicJulia/SysAdmins. 

### 📔 Set Up GitHub Pages (Public Repos Only)

1. Follow the Usage steps above to set up a new template, make sure all initial GitHub Actions have passed
2. Navigate to the repository settings and go to "Code and automation", "Pages"
3. Make sure the "Source" dropdown is set to "Deploy from a branch"
4. Set the "Branch" dropdown to "gh-pages", make sure the folder is set to "/ (root)", and click "Save"
5. Go back to the main page of your repository and click the gear to the right of the "About" section in the right side column
6. Under "Website" check the checkbox that says "Use your GitHub Pages website" and click "Save changes"
7. You will now see a URL in the "About" section that will link to your package's documentation

### 🛡️ Set Up Branch Protection (Public Repos Only)

1. Follow the Usage steps above to set up a new template, make sure all initial GitHub Actions have passed
2. Navigate to the repository settings and go to "Code and automation", "Branches"
3. Click "Add branch protection rule" to start adding branch protection
4. Under "Branch name pattern" put `main`, this will add protection to the main branch
5. Make sure to set the following options:
   - Check the "Require a pull request before merging"
   - Check the "Request status checks to pass before merging" and make sure the following status checks are added to the required list:
     - CI / Documentation
     - CI / Julia 1 - ubuntu-latest - x64 - push
     - CI / Julia 1 - ubuntu-latest - x86 - push
     - CI / Julia 1 - windows-latest - x64 - push
     - CI / Julia 1 - windows-latest - x86 - push
     - CI / Julia 1 - macOS-latest - x64 - push
   - Check the "Restrict who can push to matching branches" and add `algebraicjuliabot` to the list of people with push access
6. Click "Save changes" to enable the branch protection
