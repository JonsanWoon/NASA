<div class="modal fade" id="modalLoginForm">
    <div class="modal-dialog" role="document">
        <div class="modal-content">
            <div class="modal-header text-center">
                <h4 class="modal-title w-100 font-weight-bold">Administrator</h4>
                <button type="button" onclick="initialize()"class="close" data-dismiss="modal">&times</button>
            </div>
            <div class="modal-body mx-3">
                <div class="md-form">
                    <label>Password :</label>
                    <input id="password" type="password" class="form-control validate" autofocus>
                    <h6 id="error" class="text-danger mt-2">*Wrong password</h6>
                </div>
            </div>
            <div class="modal-footer justify-content-center">
                <button onclick="vertify()" id="lgnBtn" class="btn btn-default">Login</button>
            </div>
        </div>
    </div>
</div>
<script>
  $("#error").css("visibility", "hidden");
  function vertify(){
    var password = document.getElementById('password');
      $.ajax({
        url: 'validation',
        type: 'GET',
        data:{
          newPassword:password.value
        },
        success: function(result){
          if(result == 'true'){
            $("#modalLoginForm").modal('toggle');
             window.location.href = "settings"
             $("#error").css("visibility", "hidden");
          }
          else{
            $("#error").css("visibility", "visible");
            $("input").focus();
              password.value = "";

          }
        },
        error: function(jqXHR,exception){
          if (jqXHR.status === 0) {
                alert('Not connect.\n Verify Network.');
            } else if (jqXHR.status == 404) {
                alert('Requested page not found. [404]');
            } else if (jqXHR.status == 500) {
                alert('Internal Server Error [500].');
            } else if (exception === 'parsererror') {
                alert('Requested JSON parse failed.');
            } else if (exception === 'timeout') {
                alert('Time out error.');
            } else if (exception === 'abort') {
                alert('Ajax request aborted.');
            } else {
                alert('Uncaught Error.\n' + jqXHR.responseText);
            }
        }
      });
  }

  $('#password').keyup(function(event){
    if(event.keyCode === 13){
      $('#lgnBtn').click();
    }
  })

  function initialize(){
    $("#error").css("visibility", "hidden");
  }
</script>
