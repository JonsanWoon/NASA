<?php

namespace App\Http\Controllers;
use App\users;
use App\settings;
use Illuminate\Http\Request;
use DB;

class HomeController extends Controller
{

  private $aNews;

  public function getNews(Request $request){
    date_default_timezone_set("Asia/Kuala_Lumpur");
    $currentData = users::all();
    if(count($currentData) == 0){
      $user = new users;
      $user->userId = $request->currentUser;
      $user->userTimeIn = date('Y-m-d G:i:s');
      $initialUser = DB::table('news')->where('Result', NULL)->first();
      $user->newsId = $initialUser->NewsId;
      $user->newsTimeIn = date('Y-m-d G:i:s');
      $user->save();
    }else if(count($currentData) > 0){
      if(users::find($request->currentUser)){
      }else{
        $user = new users;
        $user->userId = $request->currentUser;
        $user->userTimeIn = date('Y-m-d G:i:s');
        $user->save();
      }
    }

    $news = DB::select('CALL getNews(?)', array($request->currentUser));
    if($news != NULL && $news[0]->Content != NULL){
      $this->aNews = $news[0];
    } else {
      $this->aNews = (object)array(
        "NewsId" => "0",
        "Headline" => "Sorry There is no news to label...",
        "Content"=> "sorry for inconvient.."
      );
    }

    $getNewsId = json_encode($this->aNews);
    return $getNewsId;
  }

  public function getSettingsItem(){
    $timeToEnableNxtBtn = settings::find('enableNextBtn');
    return $timeToEnableNxtBtn->Value;
  }

  public function storeSentiment(Request $request){
    DB::select('CALL updateNews(?,?)', array($request->sentiment,$request->newsId));
    $newRequest = new \Illuminate\Http\Request();
    $newRequest->replace(['currentUser' => $request->currentUser]);
    $this->getNews($newRequest);
    $theNews = json_encode($this->aNews);
    return $theNews;
  }

  public function checkUserTimeOut(Request $request){
    DB::select('CALL checkUserTimeOut(@status,?)', array($request->currentUser));
    $updatedRow = DB::select('select @status as status');

    if($updatedRow[0]->status == '0'){
      return 'false';
    }
    else{
      return 'true';
    }
  }
}
