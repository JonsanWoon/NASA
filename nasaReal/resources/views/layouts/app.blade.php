<html>
    <head lang="en">
        @include('includes.head')
        @yield('stylesheet')
        @yield('jsScript')
        <title>@yield('title')</title>
    </head>

    <body>
        @include('includes.header')
        @include('includes.login')
            @yield('content')
        </div>
        @include('includes.footer')
    </body>

@yield('js')
<script>
</script>
</html>
