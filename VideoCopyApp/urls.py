from django.urls import path

from . import views

urlpatterns = [path("index.html", views.index, name="index"),
		     path("UserLogin.html", views.UserLogin, name="UserLogin"),
		     path("UserLoginAction", views.UserLoginAction, name="UserLoginAction"),
		     path("Register.html", views.Register, name="Register"),
		     path("RegisterAction", views.RegisterAction, name="RegisterAction"),
		     path("UploadVideo.html", views.UploadVideo, name="UploadVideo"),
		     path("UploadVideoAction", views.UploadVideoAction, name="UploadVideoAction"),
		     path("DownloadVideo", views.DownloadVideo, name="DownloadVideo"),
		     path("DownloadVideoAction", views.DownloadVideoAction, name="DownloadVideoAction"),
		     	     
		    ]