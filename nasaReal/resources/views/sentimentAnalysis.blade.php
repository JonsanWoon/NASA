@extends('layouts.app')
@section('stylesheet')
  <link rel="stylesheet" href='{{URL::asset("css/button.css")}}'>
  <link rel="stylesheet" href='{{URL::asset("css/loader.css")}}'>
  <style>
  .border-3 {
      border-width:2px !important;
  }
  </style>
@section('title', 'NASA-Sentiment Analysis Page')
@section('content')
    <div class="row mt-2">
      <div class="col-lg-2 col-md-2 col-sm-2 col-2"></div>
      <div class="col-lg-8 col-md-8 col-sm-8 col-8">
        <textarea class="form-control" id="pastedNews" type="text" rows=15  style="resize:none;" ></textarea>
      </div>
    </div>
    <div>
      @include('includes.loader')
    </div>
    <div class="row mt-4 mx-auto">
      <div class="col-lg-2 col-md-2 col-sm-2 col-2"></div>
      <div class="col-lg-1 col-md-1 col-sm-1 col-1">
        <button  onclick="checkBtn()" id="checkBtn" class="btn btn-responsive btn-primary btn-lg shadow">Check</button>
      </div>
      <div class="col-lg-2 col-md-2 col-sm-2 col-2"></div>
      <div class="col-lg-2 col-md-2 col-sm-2 col-2">
        <label id="sentimentResult" class="text-success text-center w-100 mt-2"> Positive </label>
      </div>
      <div class="col-lg-2 col-md-2 col-sm-2 col-2"></div>
      <div class="col-lg-1 col-md-1 col-sm-1 col-1">
        <button  onclick="clearBtn()" id="clearBtn" class="btn btn-responsive btn-secondary btn-lg shadow">Clear</button>
      </div>
    </div>
      @stop

@section('js')
<script type="text/javascript" src="{{ URL::asset('js/sentimentAnalysis.js') }}"></script>
@endsection
