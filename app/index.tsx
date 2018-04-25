import * as React from 'react'
import { render } from 'react-dom'
import './app.css'


class MyClass extends React.Component<any, any> {
    constructor(props){
        super(props)
        this.state={liked:false,text:''}
        this.pclic=this.pclic.bind(this)
    }

    pclic(e){
        
       this.setState({liked:!this.state.liked});      
      
       console.log(this.state.liked);
       
       if (this.state.licked){
           console.log(this.state.licked)
           this.setState({text:'喜欢'});
       }else{
        this.setState({text:'不喜欢'});

       }
       console.log(this.state.text);
       
    }
    
    render() {
        
        return <div className="name"><button onClick={this.pclic}>hello {this.props.name}</button>{this.state.text}</div>;  
    }
}
  
  render(
    <MyClass name="fdfd" />,
    document.getElementById('app')
  );

 


//npm run dev