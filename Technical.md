# Technical setup

We now go by detailing how to setup your enviroment to start testing and studying.

We will be using a software stack based on Python 3.x with:
* Jupyter notebooks with IPython for experiment description support.
* A bunch of data handling, processing and visualization libraries.
* Deep learning libraries.

There are three ways: the local Python setup, the docker image/container and the online way.

## Setting up Python in your computer.

If your are going after this way, I recommend you to install [Anaconda's Python distribution](https://www.anaconda.com/download/) that corresponds to your platform. Anaconda comes with a much of the libraries you are going to need. Remember that if you want to try some minimally advanced deep learning stuff you need a GPU.

Anaconda installs `conda`, a command line tool for installing/updating packages. There is a GUI for this that I haven't tried.

## Using a docker image

I have updated a docker image to set up a common deep learning environment for running and testing the algorithms we are studying.

The image is extends tensorflow's. It is a Ubuntu Linux with CUDA, cuDNN, and the latest versions of tensorflow, theano, keras, etc. installed.

### How to install and use the docker image

1. Install docker: https://www.docker.com/community-edition
2. If you are a GUI person, check out [Kitematic](https://kitematic.com).
2. If you have a GPU  and Linux on your computer you should use to install nvidia-docker: https://github.com/NVIDIA/nvidia-docker

*Note:* Check https://github.com/aamini/introtodeeplearning_labs/blob/master/WindowsDocker.md for detailed instructions.

To create an instance of the image you must Run the command:
```bash
$ docker run -it -p 5678:8888 -v /home/lmarti/git/my-learning-folder:/notebooks/learning lmarti/dl
```
or, if you have nvidia-docker,
```bash
$ nvidia-docker run -it -p 5678:8888 -v /home/lmarti/git/my-learning-folder:/notebooks/learning lmarti/dl
```

* This command associates the TCP/IP 5678 port of the host (your computer) with the port 8888 of the docker vm. Jupyter notebooks run on port 8888, that is why we do this.
* It also plugs the folder /home/lmarti/git/my-learning-folder of the host as the folder /notebooks/learning of the vm.


If it runs ok you should get a log like this:

```
[I 17:49:41.905 NotebookApp] Writing notebook server cookie secret to /root/.local/share/jupyter/runtime/notebook_cookie_secret
[W 17:49:42.042 NotebookApp] WARNING: The notebook server is listening on all IP addresses and not using encryption. This is not recommended.
[I 17:49:42.071 NotebookApp] Serving notebooks from local directory: /notebooks
[I 17:49:42.072 NotebookApp] 0 active kernels
[I 17:49:42.072 NotebookApp] The Jupyter Notebook is running at:
[I 17:49:42.073 NotebookApp] http://[all ip addresses on your system]:8888/?token=245233e46f92e7092387f23ead6aa2f01d2f20611336a2be
[I 17:49:42.073 NotebookApp] Use Control-C to stop this server and shut down all kernels (twice to skip confirmation).
[C 17:49:42.074 NotebookApp]
     Copy/paste this URL into your browser when you connect for the first time,
     to login with a token:
        http://localhost:8888/?token=245233e46f92e7092387f23ead6aa2f01d2f20611336a2be
```

Now you can open a web browser and point it to http://localhost:5678 and you will be connected to your docker. You will need to copy/paste that long token you see in the dump. This can be removed if you want.

If you are running the image in a cloud service (MS Azure) you might need to create an SSH tunnel.

Remember that any changes that you do inside the container will be lost between VM reboots, except if you store it in the shared folder.

## Online services

* Jupyter notebooks can be statically rendered online on https://nbviewer.jupyter.org
* To run an online Jupyter notebook use [binder](https://mybinder.org)
* [Google Colaboratory](https://colab.research.google.com), a free online Jupyter notebook-based enviroment. To my surprise they let you use a NVidia Tesla K80 GPU for free with a 12h limit.
  On their own words:
  > Colaboratory is a Google research project created to help disseminate machine learning education and research. It's a Jupyter notebook environment that requires no setup to use and runs entirely in the cloud.

  > Colaboratory notebooks are stored in [Google Drive](https://drive.google.com) and can be shared just as you would with Google Docs or Sheets. Colaboratory is free to use.
