<?php

namespace App;

use Illuminate\Database\Eloquent\Model;

class settings extends Model
{
    public $timestamps = false;
    protected $primaryKey = 'name';
    protected $fillable = [
      'name',
      'value',
      'updatedTime',
    ];


}
