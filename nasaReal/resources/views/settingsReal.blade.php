<?php
use App\Common;
?>
@extends('layouts.app')
@section('title', 'NASA-Home Page')
@section('stylesheet')
  @parent
  <link rel="stylesheet" href="{{URL::asset('css/loader.css')}}">
  <link rel="stylesheet" href="{{URL::asset('css/fieldset.css')}}">
    <style>
      label{
        font-size: 20px;
        margin-right:10px;]
      }
    </style>
    @stop
@section('content')
@if(session()->has('message'))
      <script> alert('Settings Saved !')</script>
@endif
<div class="row">
  <div class="col-lg-1"></div>
    <div class="col-lg-10">
  {!! Form::model($settings,[
      'route' => ['settings.update'],
      'class' => 'form-inline'
    ])!!}

    <fieldset class="fieldset">
      <legend  class="legend">
        General Settings
      </legend>
    <div class="row">
        {!! Form::label('settings-OldPassword', 'Old Password:', [
          'class' => 'col-lg-3 col-md-3 col-sm-3 ',
        ])!!}
      <div class="col-lg-3 col-md-3 col-sm-3">
        {!! Form::password('OldPassword', [
          'id' => 'oldPassword',
          'class' => 'form-control',
        ])!!}
        @if ($errors->has('OldPassword'))
          <span class="text-danger">*wrong password</span>
        @endif
        @if ($errors->has('OldPasswordRequired'))
        <span class="text-danger">*required</span>
        @endif
      </div>

        {!! Form::label('settings-Timetoenablenextbutton', 'Time to enable next button:', [
          'class' => 'col-lg-3 col-md-3 col-sm-3',
        ])!!}
      <div class="col-lg-2 col-md-2 col-sm-2">
        {!! Form::number(
          'Timetoenablenextbutton', $settings->enableNextBtn, [
          'min' => '0',
          'max' => '60',
          'id' => 'timetoenablenextbutton',
          'class' => 'form-control w-50',
        ])!!}
        @if ($errors->has('Timetoenablenextbutton'))
        <span class="text-danger">*required</span>
        @endif
      </div>
    </div>
    <div class="row">
          {!! Form::label('settings-NewPassword', 'New Password:', [
            'class' => 'col-lg-3 col-md-3 col-sm-3 mt-3',
          ])!!}
        <div class="col-lg-3 col-md-2 col-sm-2">
          {!! Form::password('NewPassword', [
            'id' => 'newPassword',
            'class' => 'form-control mt-3',
          ])!!}
          @if ($errors->has('NewPassword'))
          <span class="text-danger">*required</span>
          @endif
        </div>
        {!! Form::label('settings-housekeep', 'Housekeep:', [
          'class' => 'col-lg-3 col-md-3 col-sm-3 ',
        ])!!}
      <div class="col-lg-2 col-md-3 col-sm-3">
        {!! Form::number('Housekeep', $settings->housekeep, [
          'min' => '0',
          'id' => 'housekeep',
          'class' => 'form-control w-50',
        ])!!}
        @if ($errors->has('Housekeep'))
        <span class="text-danger">*required</span>
        @endif
      </div>
    </fieldset>

    <fieldset class="fieldset">
      <legend  class="legend">
        Backend Parameter
      </legend>
    <div class="row">
        {!! Form::label('settings-Url', 'Url:', [
          'class' => 'col-lg-3 col-md-3 col-sm-3 ',
        ])!!}
      <div class="col-lg-3 col-md-3 col-sm-3">
        {!! Form::text('Url', $settings->url, [
          'id' => 'url',
          'class' => 'form-control w-100',
        ])!!}
        @if ($errors->has('Url'))
        <span class="text-danger">*required</span>
        @endif
        @if ($errors->has('errorFormat'))
          <span class="text-danger">*format error</span>
        @endif
      </div>

        {!! Form::label('settings-Port', 'Port:', [
          'class' => 'col-lg-3 col-md-3 col-sm-3',
        ])!!}
      <div class="col-lg-2 col-md-3 col-sm-3">
        {!! Form::number('Port', $settings->port, [
          'min' => '0',
          'max' => '65535',
          'id' => 'port',
          'class' => 'form-control w-75',
        ])!!}
        @if ($errors->has('Port'))
        <span class="text-danger">*required</span>
        @endif
      </div>
    </div>
<div class="row">
    {!! Form::label('settings-RandomState', 'Random state:', [
      'class' => 'col-lg-3 col-md-3 col-sm-3 mt-2',
    ])!!}
  <div class="col-lg-3 col-md-3 col-sm-3 mt-2">
    {!! Form::number('RandomState', $settings->randomstate, [
      'min' => '0',
      'id' => 'randomState',
      'class' => 'form-control w-50',
    ])!!}
    @if ($errors->has('RandomState'))
      <span class="text-danger">*required</span>
    @endif
  </div>
  {!! Form::label('settings-nb_epoch', 'Number of epoch:', [
    'class' => 'col-lg-3 col-md-3 col-sm-3 mt-2',
  ])!!}
<div class="col-lg-2 col-md-3 col-sm-3 mt-2">
  {!! Form::number('nb_epoch', $settings->nb_epoch, [
    'min' => '0',
    'id' => 'nb_epoch',
    'class' => 'form-control w-50',
  ])!!}
  @if ($errors->has('nb_epoch'))
  <span class="text-danger">*required</span>
  @endif
</div>
</div>
<div class="row">
    {!! Form::label('settings-trainVerbose', 'Train verbose:', [
      'class' => 'col-lg-3 col-md-3 col-sm-3 mt-2',
    ])!!}
  <div class="col-lg-3 col-md-3 col-sm-3 mt-2">
    {!! Form::number('TrainVerbose', $settings->train_verbose, [
      'min' => '0',
      'id' => 'trainVerbose',
      'class' => 'form-control w-50',
    ])!!}
    @if ($errors->has('TrainVerbose'))
    <span class="text-danger">*required</span>
    @endif
  </div>
  {!! Form::label('settings-evaluateVerbose', 'Evaluate verbose:', [
    'class' => 'col-lg-3 col-md-3 col-sm-3 mt-2',
  ])!!}
<div class="col-lg-2 col-md-3 col-sm-3 mt-2">
  {!! Form::number('EvaluateVerbose', $settings->evaluate_verbose, [
    'min' => '0',
    'id' => 'evaluateVerbose',
    'class' => 'form-control w-50',
  ])!!}
  @if ($errors->has('EvaluateVerbose'))
  <span class="text-danger">*required</span>
  @endif
</div>
</div>
    <div class="row">
          {!! Form::label('settings-RmvCommonWord', 'Remove Common Word:', [
            'class' => 'col-lg-3 col-md-3 col-sm-3 mt-2',
          ])!!}
        <div class="col-lg-3 col-md-3 col-sm-3 mt-2">
          {!! Form::radio('RmvComWord', 'true', $settings->rmvComWordT,[
            'value' => 'true',
            'id' => 'rmvCommonWordY',
            'class' => 'form-control ',
          ])!!}
          Yes
          {!! Form::radio('RmvComWord', 'false', $settings->rmvComWordF,[
            'value' => 'false',
            'id' => 'rmvCommonWordN',
            'class' => 'form-control ',
          ])!!}
          No
        </div>
      </div>
    <div class="row">
      {!! Form::label('settings-StemmingWord', 'Stemming Word:', [
        'class' => 'col-lg-3 col-md-3 col-sm-3 mt-2',
      ])!!}
        <div class="col-lg-3 col-md-3 col-sm-3 mt-2">
          {!! Form::radio('StemmingWord', 'true' , $settings->stemmingWordT, [
            'value' => 'true',
            'id' => 'stemmingWordY',
            'class' => 'form-control',
          ])!!}
          Yes
          {!! Form::radio('StemmingWord', 'false' , $settings->stemmingWordF, [
            'value' => 'false',
            'id' => 'stemmingWordN',
            'class' => 'form-control ',
          ])!!}
          No
        </div>
      </div>
    </fieldset>

        <div class="col-lg-5 col-md-5 col-sm-5 col-5"></div>
        <div class="col-lg-2 col-md-2 col-sm-2 col-2">
      {!! Form::button('Save', [
        'type' => 'submit',
        'class' => 'btn btn-primary btn-lg mt-4 w-100 shadow'
     ]) !!}
       </div>
      {!! Form::close() !!}
    </div>
  </div>
@endsection
