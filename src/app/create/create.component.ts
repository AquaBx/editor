import { Component } from '@angular/core';
import { CommonModule } from '@angular/common';
import { RouterOutlet } from '@angular/router';

import { SelectBox } from '../components/SelectBox.component'

@Component({
  standalone: true,
  imports: [CommonModule, RouterOutlet,SelectBox],
  templateUrl: './create.component.html',
  styleUrl: './create.component.css'
})

export class CreateView {

}
