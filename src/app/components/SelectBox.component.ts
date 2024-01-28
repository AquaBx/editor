import { Component, Input } from '@angular/core';
import { CommonModule } from '@angular/common';

@Component({
    selector:"SelectBox",
    standalone: true,
    imports: [CommonModule],
    template:`
    <div class="dropbox">
        <button>{{options[selected]}}</button>
        <ul class="fold">
            <li *ngFor='let option of options' (click)='change(option)'>{{option}}</li>
        </ul>
    </div>
    `,
    styles:`
    .dropbox{
        position:relative;
        width:fit-content;
    }
    .dropbox:focus-within > .fold{
        display:block;
    }

    .fold {
        display:none;
        position:absolute;
        flex-direction:column;
        height:300px;
        padding:0;
        width:100%;
    }

    button,li{
        all:unset;
        cursor:pointer;
        display:flex;
        gap:16px;
        align-items:center;
        background-color:cyan;
        padding:10px 16px;
        width:100%;
        justify-content: space-between;
        background:#f0f0f0;
    }

    button:hover,li:hover {
        background:#eeeeff;
    }

    button::after {
        font-size:10px;
        content:"▼";
    }

    li {
        display:block;
        width:100%;
    }

    `
})

export class SelectBox {
    @Input()
    options:string[] = ["Template 1","Template 2"]

    selected = 0;

    change(i:string){
        console.log(i)
    }
}
