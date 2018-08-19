var sentiment = "NULL";
var newsId = "NULL";
var currentUser = "NULL";
var textarea = document.getElementById('content');
var timeToEnableNextBtn = 0;
$("#loader").css("display", "none");

$(document).ready(function(){
  var guid = (S4() + S4() + "-" + S4() + "-4" + S4().substr(0,3) + "-" + S4() + "-" + S4() + S4() + S4()).toLowerCase();
  currentUser = guid;
  getNews(guid);
});

function S4() {
    return (((1+Math.random())*0x10000)|0).toString(16).substring(1);
}

function getSettings(){
  $.ajax({
    url: 'getSettingsItem',
    type: 'GET',
    data: {none:"none"},
    success: function(settings){
      timeToEnableNextBtn = parseInt(settings);
    },
    error: function(jqXHR,exception){
      checkingError(jqXHR.status, exception);
    }
  });
}

function getNews(userId){
  $.ajax({
    url: 'getNews',
    type: 'GET',
    data: {currentUser: userId},
    success: function(news){
      var theNews = JSON.parse(news);
      newsId = theNews.NewsId;

      textarea.value = theNews.Content;
      document.getElementById('headline').innerHTML = theNews.Headline;

      if(theNews.NewsId == "0"){
        disable();
      }else{
        getSettings();
        doCountDown();
        $('#nxtBtn').attr('disabled', 'true');
      }
    },
    error: function(jqXHR,exception){
      checkingError(jqXHR, exception);
    }
  });
}

function doCountDown(){
  setTimeout(function(){
    countDown(timeToEnableNextBtn);
  },1000);
}
function countDown(i) {
    var interval = setInterval(function () {
      if(i == 0){
        clearInterval(interval);
        $("#nxtBtn").removeAttr("disabled");
        document.getElementById("nxtBtn").innerHTML = "Next";
      }else{
        document.getElementById("nxtBtn").innerHTML = "Next(" + i +")";
        i-- || clearInterval(interval);
      }
    }, 1000);
}

function toastErrorMessage() {
    var error = document.getElementById("toast");
    error.className = "show";
    setTimeout(function(){
      error.className = error.className.replace("show", "");
    }, 3000);
}

$("#ngtBtn").click(function(){
    sentiment = displaySentimentSelected(-1);

});
$("#nutBtn").click(function(){
    sentiment = displaySentimentSelected(0);
});
$("#pstBtn").click(function(){
    sentiment = displaySentimentSelected(1);
});
$("#nxtBtn").click(function(){

    if(sentiment == "NULL"){
      toastErrorMessage();
    }
    else{
      disable();
      setTimeout(function(){
        checkUserTimeOut();
        setTimeout(function(){
          enable();
        }, 300);
      }, 800);
   }


});

function displaySentimentSelected(value){
  if(value == 0){
    $("#content").removeClass("border-3 border-danger border-success");
    $("#content").addClass("border-3 border-primary");
  }else if(value == 1){
    $("#content").removeClass("border-3 border-primary border-danger");
    $("#content").addClass("border-3 border-success");
  }else{
    $("#content").removeClass("border-3 border-primary border-success");
    $("#content").addClass("border-3 border-danger");
  }
  return value;
}
function enable(){
  $("#pstBtn").removeAttr("disabled");
  $("#ngtBtn").removeAttr("disabled");
  $("#nxtBtn").removeAttr("disabled");
  $("#nutBtn").removeAttr("disabled");
  $("#loader").css("display","none");
}

function disable(){
  $("#pstBtn").attr("disabled","true");
  $("#ngtBtn").attr("disabled","true");
  $("#nxtBtn").attr("disabled","true");
  $("#nutBtn").attr("disabled","true");
  $("#loader").css("display","block");
}

function checkUserTimeOut(){
  $.ajax({
   url: 'checkUserTimeOut',
   type: 'GET',
   data: {
     currentUser:currentUser
   },
   success: function(userTimeOut){
     //alert(userTimeOut);
     if(userTimeOut == 'true'){
       disable();
       alert('Session has been expired. Please refresh the page.');
     }else if(userTimeOut == 'false'){
       storeSentiment();
     }
   },
   error: function(jqXHR,exception){
     checkingError(jqXHR, exception);
   }
 });
}

function storeSentiment(){
  $.ajax({
   url: 'storeSentiment',
   type: 'POST',
   data: {
     sentiment: sentiment,
     newsId: newsId,
     currentUser: currentUser
   },
   success: function(news){
     $("#content").removeClass("border-3 border-primary border-success border-danger");
     sentiment = "NULL";
     var theNews =  JSON.parse(news);
     newsId = theNews.NewsId;
     textarea.style.backgroundColor = "";
     textarea.value = theNews.Content;
     document.getElementById('headline').innerHTML = theNews.Headline;
     $('#nxtBtn').attr('disabled', 'true');
     doCountDown();

   },
   error: function(jqXHR,exception){
     checkingError(jqXHR, exception);
   }
 });
}

function checkingError(errorMessage, exception){
  if (errorMessage.status === 0) {
        alert('Not connect.\n Verify Network.');
    } else if (errorMessage.status == 404) {
        alert('Requested page not found. [404]');
    } else if (errorMessage.status == 500) {
        alert('Internal Server Error [500].');
    } else if (exception === 'parsererror') {
        alert('Requested JSON parse failed.');
    } else if (exception === 'timeout') {
        alert('Time out error.');
    } else if (exception === 'abort') {
        alert('Ajax request aborted.');
    } else {
        alert('Uncaught Error.\n' + errorMessage.responseText);
    }
}
