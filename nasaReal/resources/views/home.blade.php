@extends('layouts.app')
@section('stylesheet')
  <link rel="stylesheet" href="{{URL::asset('css/button.css')}}">
  <link rel="stylesheet" href="{{URL::asset('css/loader.css')}}">
  <link rel="stylesheet" href="{{URL::asset('css/toast.css')}}">
  <style>
  .border-3 {
      border-width:2px !important;
  }
  </style>
@section('jsScript')
@section('title', 'NASA-Home Page')
@section('content')
      <div id="toast">Select either one sentiment to label the news</div>
      <div class="row">
        <div class="col-lg-2 col-md-2 col-sm-2 col-2"></div>
        <div id="headline" class="col-lg-8 col-md-8 col-sm-8 col-8 h3">
        </div>
      </div>
      <div class="row">
        <div class="col-lg-2 col-md-2 col-sm-2 col-2"></div>
        <div class="col-lg-8 col-md-8 col-sm-8 col-8">
          <textarea id="content" class="form-control" type="text" rows=15  style="resize:none" readonly>
          </textarea>
        </div>
      </div>
      <div class="row mt-3">
        <div class="col-lg-3 col-md-3 col-sm-3 col-3"></div>
        <div class="col-lg-1 col-md-1 col-sm-1 col-1">
          <button id="ngtBtn"  class="label btn btn-responsive btn-danger btn-md shadow-lg">Negative</button>
        </div>
        <div class="col-lg-1 col-md-1 col-sm-1 col-1">
          <button id="nutBtn"  class="label btn btn-responsive btn-primary btn-md shadow-lg">Neutral</button>
        </div>
        <div class="col-lg-1 col-md-1 col-sm-1 col-1">
          <button id="pstBtn"  class="label btn btn-responsive btn-success btn-md shadow-lg">Positive</button>
        </div>
        <div class="col-lg-2 col-md-2 col-sm-2 col-2"></div>
        <div class="col-lg-1 col-md-1 col-sm-1 col-1">
          <button id="nxtBtn" class="label btn btn-responsive btn-secondary btn-md shadow">Next</button>
        </div>
      </div>
      <div>
        @include('includes.loader')
      </div>
@stop
@section('js')
<script type="text/javascript" src="{{ URL::asset('js/home.js') }}"></script>
@endsection
