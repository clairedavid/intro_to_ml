(tuto:setup)=
# Setup

The tutorials will be done on Google Colab, a remote Python server running in the cloud. Although you access it from your account, Colab runs on Google’s servers and does not automatically have access to the files on your local computer or Google Drive.

You will be given data files to download and upload to your Google Drive. Storing files in Drive is the easiest way to make them accessible to Colab without extra setup. But how do you access them from Colab? This section will guide you through the steps.


(bash:refresher)=
## Bash (re)fresher

You will need to be familiar with the basics of bash for your setup mission.

````{margin}
<br><br><sup>*</sup> Bash (Bourne Again SHell) is a programming language for Unix-like operative systems. It's an interactive command-interpreter, that means it allows the user to "talk" to the operating system (OS) and execute tasks. Thanks to scripts written in bash, tasks can be automatized. 

````
```{admonition} Note to the Unix whisperers
:class: note
If you are familiar with bash<sup>*</sup>, you can quickly go through this. If not, follow along and learn the basic commands that will help you to navigate in a terminal. 
```



Here are the most common commands in bash: 

| Command   | Meaning / Action                          | Example / Usage                          |
|-----------|-------------------------------------------|------------------------------------------|
| `ls`      | List files and folders                    | `ls`                                     |
| `pwd`     | Print working directory (where you are)   | `pwd`                                    |
| `cd`      | Change directory                          | `cd myfolder/`                           |
| `cd ..`   | Go up one directory                       | `cd ..`                                  |
| `mkdir`   | Make a new directory (folder)             | `mkdir a_new_folder`                     |
| `rm`      | ⚠️ Remove a file ⚠️                       | `rm a_file.txt`                          |
| `rm -r`   | ⚠️ Remove a directory (recursively) ⚠️    | `rm -r a_folder/`                        |
| `cp`      | Copy files or folders                     | `cp a_file.txt copied_file.txt`          |
| `mv`      | Move or rename a file or folder           | `mv old.txt new.txt`                     |
| `cat`     | Show file contents                        | `cat notes.txt`                          |
| `head`    | Show first 5 lines of a file              | `head -n 5 a_big_file.csv`               |
| `tail`    | Show last 12 lines of a file              | `tail -n 12 another_big_file.txt`        |

<br>

In Colab, a code cell runs Python by default. To run a bash command, put a `!` in front of it. This will however work only in the current cell. To run a bash command persisting across the follow cells, you need to use the `%` sign. In brief:
* Use `!` for quick inspections like `!ls`, `!cat file.txt`, `!head file.csv`.
* Use `%cd` only if you want to change the working directory for Colab in a persistent way.


## Your setup mission

Let’s do a short exercise to get familiar with the steps that will be done at each tutorial: 
1. Download the dataset
2. Upload it to your Google Drive
3. Mount your Google Drive in Colab
4. Read the file from Colab

### 1. Download the dataset
Here is a cute little dataset:  (you will understand later)  
[Download dataset](https://drive.google.com/uc?export=download&id=1rxEFuLfbaxed8p0pKprBmgn6Izf4SYWU)

### 2. Upload the file on your Drive
Start by uploading the dataset to your Google Drive. It can help to keep things organized in a folder like `data/intro_to_ml`, but feel free to place it wherever works best for you.

### 3. Mount your Google Drive in Colab
In your Google Drive, click on &nbsp; `+ New`&nbsp;  and then &nbsp; `More`&nbsp;  and click on &nbsp; `Google Colaboratory`. 

To access your Google Drive files, you need to create a 'bridge' from the Colab server to your Drive. This is done by first importing a special library and then entering this command:
```python
from google.colab import drive
drive.mount('/content/gdrive')
```
Copy and paste the above in your blank Colab file. A popup will appear asking you to grant Colab access to your Google Drive: click "accept" to proceed.

### 4. Read your file
Now let's navigate with bash commands to find that file.

You can first do:

```bash
!pwd
```

to see where you are in your Drive and then list the file content with `ls` or to put more details, using the option `-l`: 

```bash
!ls -l 
``` 

Then navigate to the folders where your file is using a couple of `cd` and `ls` commands. For the `cd` command, remember to use the `%` symbol at front. 

```{tip}
Once you locate your data file in Google Drive, it’s a good practice to store its full path (i.e., the string of characters representing all directories and subdirectories) in a variable. You can name it `data_path` for instance. Careful using `path` as it is a special python variable! 
```

__Let's inspect the file!__  
In the section {ref}`bash:refresher` above, you can see the commands `cat`, `head` and `tail`. The first one will display in the terminal the entire file. What if there are many many lines? This will saturate your output window. Instead, you can use `head` or `tail` to display a portion of the file. For instance, `head -n 10` shows the first 10 lines, `tail -n 3` will give you the last three lines. 

In my Colab setup with Drive mounted, it looks like this:

```python
data_path = "/content/gdrive/MyDrive/data/intro_ml_2025/"
data_file = "data_setup.csv"
```
In bash, variables are prefixed with a `$` sign. To use multiple variables (or a variable and a string) together, just place them back-to-back without any separator.
```bash 
!head -n 10 $data_path$data_file
```
This will print the first 10 lines of the file.

It's important to reach this point before starting the tutorials. Ask your TA for help if need be. And remember you are here to learn! 

### Bonus: plot the data 
Curious about what this setup dataset looks like? Use this plotting macro and see for yourself 😉

```python
import os
import pandas as pd
import matplotlib.pyplot as plt

def plot_setup_data(csv_file):
    # Check if file exist
    if not os.path.exists(csv_file):
        print(f"File {csv_file} does not exist.")
        return
    
    df = pd.read_csv(csv_file)

    required_cols = {"x1", "x2", "y"}
    if not required_cols.issubset(df.columns):
        print(f"CSV must contain columns: {required_cols}")
        return
    
    # map y=0 -> black, y=1 -> red
    colors = df["y"].map({0: "black", 1: "red"})
    
    plt.figure(figsize=(6, 6))
    plt.scatter(df["x1"], df["x2"], c=colors, s=8)
    plt.xlabel(r"$x_1$"); plt.ylabel(r"$x_2$")
    
    # Square plot from 0 to 1, no other tick
    plt.xlim(0, 1); plt.ylim(0, 1)
    plt.xticks([0, 1]); plt.yticks([0, 1])
    plt.axis("equal")
    plt.grid(False)

    plt.show()
```


You're done the setup! Enjoy the tutorials! 