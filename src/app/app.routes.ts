import { Routes } from "@angular/router";
import { HomeView } from "./home/home.component";
import { CreateView } from "./create/create.component";
import { EditorView } from "./editor/editor.component";

export const routes: Routes = [
    { path: '', component: HomeView },
    { path: 'create', component: CreateView },
    { path: 'editor', component: EditorView },
];
