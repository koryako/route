/* build/webpack.config.js */
const config = {
    entry: './app/index.tsx',
    output: {
      filename: 'app.bundle.js',
      path: __dirname + 'public',
      publicPath: '/assets'
    },
    devtool: 'source-map',
    resolve: {
      extensions: ['.webpack.js', '.web.js', '.ts', '.tsx', '.js']
    },
    module: {
        rules: [
        {
          test: /\.tsx?$/,
          loader: 'ts-loader'
        },
        {
            test: /\.css/,
            loader: 'style-loader!css-loader'
          }
      ],
      //preLoaders: [
        //{
         // test: /\.js$/,
          //loader: 'source-map-loader'
        //}
      //]
    },
    devtool: 'eval'
  }
  
  module.exports = config