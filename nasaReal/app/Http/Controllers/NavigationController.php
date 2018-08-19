<?php

namespace App\Http\Controllers;

use Illuminate\Http\Request;

class NavigationController extends Controller
{
    public function settings(){
      return view('settings');
    }

    public function home(){
      return view('home');
    }
    public function sentimentAnalysis(){
      return view('sentimentAnalysis');
    }
}
