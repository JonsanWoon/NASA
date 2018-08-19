<?php

namespace App;

use Illuminate\Database\Eloquent\Model;

class users extends Model
{
    public $timestamps = false;
    protected $primaryKey = 'userId';
    protected $fillable=[
      'userId',
      'userTimeIn',
      'newsId',
      'newsTimeIn',
    ];
}
