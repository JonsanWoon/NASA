<?php

namespace App\Http\Controllers;
use App\settings;
use Illuminate\Http\Request;

class SentimentController extends Controller
{
    public function getSentimentResult(Request $request){
      $url = settings::find('url');
      $port = settings::find('port');
      $ch = curl_init($url->Value. ":" .$port->Value);
      $data = array(
          'retrain' => '0',
          'news' => '"'.$request->news.'"',
          'password' => 'password'
        );
      $options = array(
        CURLOPT_RETURNTRANSFER => true,
      );
      $jsonData = json_encode($data);

      curl_setopt_array($ch, $options);
      curl_setopt($ch, CURLOPT_POST, 1);
      curl_setopt($ch, CURLOPT_POSTFIELDS, $jsonData);
      curl_setopt($ch, CURLOPT_HTTPHEADER, array('Content-Type: application/json'));

      $result = curl_exec($ch);
      curl_close($ch);
      return $result;
    }
}
