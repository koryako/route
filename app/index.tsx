import * as React from 'react'
import { render } from 'react-dom'
import './app.css'


class MyClass extends React.Component<any, any> {
    constructor(props){
        super(props)
        this.state={liked:false}
        this.pclic=this.pclic.bind(this)
    }

    pclic(e){
        
       this.setState({liked:!this.state.liked});      
      
       console.log(this.state.liked);
    }
    
    render() {
        let text=this.state.licked? "不喜欢":"喜欢";
        return <div><button onClick={this.pclic}>hello {this.props.name}</button>{text}</div>;  
    }
}
  
  render(
    <MyClass name="fdfd" />,
    document.getElementById('app')
  );

 


//npm run dev