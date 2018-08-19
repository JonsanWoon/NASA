<?php

namespace App\Http\Controllers;
use App\settings;   //use the model that defined
use DB;

use Illuminate\Http\Request;
use Illuminate\Support\Facades\Hash;
class SettingController extends Controller
{
  //login
  public function login(){
    $settings = new settings();
    return view('login', [
      'login' => $settings->Password,
    ]);
  }

  //display the previous settings on page
  public function edit()
 {
    $enableNextBtn = settings::find('enableNextBtn');
    $url = settings::find('url');
    $port = settings::find('port');
    $randomState = settings::find('randomstate');
    $nb_epoch = settings::find('nb_epoch');
    $trainVerbose = settings::find('train_verbose');
    $evaluateVerbose = settings::find('evaluate_verbose');
    $rmvComWordT = settings::find('rmvCommonWord') ;
    $stemmingWordT = settings::find('stemmingWord');
    $rmvComWordF = settings::find('rmvCommonWord');
    $stemmingWordF = settings::find('stemmingWord');
    $housekeep = settings::find('housekeep');
    $settings = (object)array(
        "enableNextBtn" => $enableNextBtn->Value,
        "url" => $url->Value,
        "port" => $port->Value,
        "rmvComWordT" => $rmvComWordT->Value  == '1' ? true : false ,
        "stemmingWordT" => $stemmingWordT->Value  == '1' ? true : false,
        "rmvComWordF" => $rmvComWordF->Value == '0' ? true : false,
        "stemmingWordF" => $stemmingWordF->Value == '0' ? true : false,
        "randomstate" => $randomState->Value,
        "nb_epoch" => $nb_epoch->Value,
        "train_verbose" => $trainVerbose->Value,
        "evaluate_verbose" => $evaluateVerbose->Value,
        "housekeep" => $housekeep->Value
      );

     return view('settingsReal', [
       'settings' => $settings
     ]);
 }

 //update the settings
 public function update(Request $request)
 {
   $ip = $password = $url = false;
   $errors = new \Illuminate\Support\MessageBag;
   $validate = $this->validate(
     $request,[
       'Url' => 'required',
       'Port' => 'required',
       'Timetoenablenextbutton'=>'required',
       'RandomState' => 'required',
       'nb_epoch' => 'required',
       'TrainVerbose' => 'required',
       'EvaluateVerbose' => 'required'
   ]);
   $currentPassword = settings::where('name','password')->first();
   $theUrl = filter_var($request->Url, FILTER_SANITIZE_URL);

   if (filter_var($request->Url, FILTER_VALIDATE_IP)) {  //validate ip address
     $ip = true;
   } else if(filter_var($theUrl, FILTER_VALIDATE_URL)){  //validate url
     $url = true;
   } else{
     $errors->add('errorFormat', 'dummyErrorMessage');
     $ip = false;
     $url = false;
   }
   if($request->OldPassword != '' && $request->NewPassword != ""){
     if(Hash::check($request->OldPassword, $currentPassword->Value)){
       $password = true;
     }else{
       $errors->add('OldPassword', 'dummyErrorMessage');
       $password = false;
     }
   } else if($request->OldPassword != '' && $request->NewPassword == ""){
      $errors->add('NewPassword','dummyErrorMessage');
      $password = false;
   } else if($request->OldPassword == '' && $request->NewPassword != ""){
      $errors->add('OldPasswordRequired','dummyErrorMessage');
      $password = false;
   } else {
     $password = true;
   }
   if($password && ($ip == true || $url == true)){

   $getRmvComWord = $request->RmvComWord == 'true' ? '1' : '0';
   $getStemmingWord = $request->StemmingWord == 'true' ? '1' : '0';
   
   if($request->NewPassword == "" && $request->OldPassword == ""){
   }else{
     $this->savingPassword($request->NewPassword);
   }
     $this->savingOther("url", $request->Url);
     $this->savingOther("housekeep", $request->Housekeep);

   $port = $this->savingOther("port", $request->Port);
   $stemmingWord = $this->savingOther("stemmingWord", $getStemmingWord);
   $rmvComWord = $this->savingOther("rmvCommonWord", $getRmvComWord);
   $enableNextBtn = $this->savingOther("enableNextBtn", $request->Timetoenablenextbutton);
   $randomState = $this->savingOther("randomstate", $request->RandomState);
   $nb_epoch = $this->savingOther("nb_epoch", $request->nb_epoch);
   $train_verbose = $this->savingOther("train_verbose", $request->TrainVerbose);
   $evaluate_verbose = $this->savingOther("evaluate_verbose", $request->EvaluateVerbose);

           if($port || $stemmingWord || $rmvComWord || $enableNextBtn || $randomState || $nb_epoch || $train_verbose || $evaluate_verbose){
             $this->retrain();
           }else{
           }
    return redirect()->back()->with('message', 'Settings Saved!');
 }
 else{
  return back()->withErrors($errors);
 }
}

public function retrain(){
  $url = settings::find('url');
  $port = settings::find('port');
  $ch = curl_init($url->Value. ":" .$port->Value);
  $data = array(
      'retrain' => '1',
      'password' => 'password'
    );
  $jsonData = json_encode($data);

  curl_setopt($ch, CURLOPT_POST, 1);
  curl_setopt($ch, CURLOPT_POSTFIELDS, $jsonData);
  curl_setopt($ch, CURLOPT_HTTPHEADER, array('Content-Type: application/json'));

  $result = curl_exec($ch);
  curl_close($ch);
}

//perform password validation
 public function validation(Request $request){
   $currentPassword = settings::where('name','password')->first();
   if(Hash::check($request->newPassword, $currentPassword->Value)){
     return 'true';
   }else{
     return 'false';
   }
 }

 private function savingPassword($password){
   settings::where('name', 'password')
           ->update([
             'value' => Hash::make($request->NewPassword),
             'updatedTime' => date('Y-m-d G:i:s')
           ]);
 }
 private function savingOther($name, $value){
   settings::where('name', $name)
           ->update([
             'value' => $value,
             'updatedTime' => date('Y-m-d G:i:s')
           ]);
 }
}
