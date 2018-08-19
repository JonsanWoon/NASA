<?php

/*
|--------------------------------------------------------------------------
| Web Routes
|--------------------------------------------------------------------------
|
| Here is where you can register web routes for your application. These
| routes are loaded by the RouteServiceProvider within a group which
| contains the "web" middleware group. Now create something great!
|
*/

Route::get('/', function(){   //initial pages
  return View('home');
});

Route::get('settings', 'SettingController@edit');              //route to settings pages
Route::get('home', 'NavigationController@home');                             //route to home pages
Route::get('sentimentAnalysis', 'NavigationController@sentimentAnalysis');   //route to sentiment sentiment analysis pages

Route::post('getResult','SentimentController@getResult');              //route to function getresult() in the sentiment controller
Route::get('getNews', 'HomeController@getNews');               //route to function getNews() get new to display
Route::get('getSettingsItem', 'HomeController@getSettingsItem');
Route::post('getSentimentResult', 'SentimentController@getSentimentResult');
Route::post('storeSentiment','HomeController@storeSentiment');               /*route to functions storeSentiment()
                                                                               store the sentiment label*/
Route::get('login', 'SettingController@login');
Route::get('validation', 'SettingController@validation');
Route::post('update', 'SettingController@update')->name('settings.update');      //route to function to do update settings
Route::get('checkUserTimeOut','HomeController@checkUserTimeOut');
