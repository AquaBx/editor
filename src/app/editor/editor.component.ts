import { Component } from '@angular/core';
import { CommonModule } from '@angular/common';
import { RouterOutlet } from '@angular/router';

import { invoke } from '@tauri-apps/api';

@Component({
  standalone: true,
  imports: [CommonModule, RouterOutlet],
  templateUrl: './editor.component.html',
  styleUrl: './editor.component.css'
})

export class EditorView {
  async render() {
    let img:{buffer:number[],width:number,height:number} = await invoke('my_typst_render')

    var canvas = document.createElement('canvas')
    var ctx = canvas.getContext('2d')!
    canvas.width = img.width;
    canvas.height = img.height;

    // create imageData object
    var idata = ctx.createImageData(img.width, img.height);

    // set our buffer as source
    idata.data.set(img.buffer);

    // update canvas with new data
    ctx.putImageData(idata, 0, 0);

    console.log(canvas.toDataURL())
    // console.log(buffer)
  }
}
